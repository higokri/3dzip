import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize, adaptive_voxelize
from torch_scatter import scatter_mean, scatter_add
from .position_encodings import PositionEmbeddingLearnedMLP


class PromptEncoder(nn.Module):

    def __init__(self, latent_dim=4096):
        super(PromptEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.pos_emb3d = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.pos_emb3d(xyz)

    def forward(self, clicks):
        pos_embed = self.encode_pe(clicks)
        return pos_embed


class RGBDVideoTower(nn.Module):
    def __init__(self, vision_tower, video_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.num_frames = args.num_frames
        self.num_sample_tokens = args.num_sample_tokens
        self.pooling = os.environ.get('POOLING', 'voxelize')
        self.voxel_size = float(os.environ.get('VOXEL_SIZE', '0.2'))
        self.adaptive_ratio = float(os.environ.get('ADAPTIVE_RATIO', '0'))
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            return

        self.video_processor = RGBDVideoProcessor(self.vision_tower_name, self.num_frames)
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule()
        else:
            raise NotImplementedError

        self.prompt_encoder = PromptEncoder()
        self.is_loaded = True

    def forward(self, features, depths, poses, intrinsics, lengths=None, cls_attn=None, text_sim=None):
        """
        Args:
            - features: (B, V, 1024, 24, 24), image token features
            - depths: (B, V, H, W), depth images
            - poses: (B, V, 4, 4), camera-to-world poses
            - intrinsics: (B, V, 4, 4), camera intrinsics
            - lengths: (B,), view count per scene

        Returns:
            - pooled_video_features: (Bn, 1024)
            - batch_offset: (B,) int32
        """
        B, V, C, H, W = features.shape
        _log = not getattr(self, '_logged_once', False)
        assert intrinsics.dim() == 4

        # Depth backprojection -> 3D world coordinates
        feat_xyz, xyz = backprojector_dataloader([features.flatten(0, 1)], depths, poses, intrinsics)
        # Spatial-aware 3D position embedding
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        video_xyz = feat_xyz.reshape(B, V * H * W, 3)
        if lengths is not None:
            lengths = lengths * H * W

        # Adaptive voxel size: find voxel_size that yields ~target tokens
        if self.adaptive_ratio > 0:
            target = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else int(self.adaptive_ratio * V * H * W)
            p2v, found_vs = adaptive_voxelize(feat_xyz, target_tokens=target)
        else:
            p2v = None

        if self.pooling == 'voxelize':
            # Baseline: scatter_mean per voxel grid
            if p2v is None:
                p2v = voxelize(feat_xyz, self.voxel_size)
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(B)])
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)

        elif self.pooling == '3dzip':
            # 3DZip: Voxelize + DPP greedy MAP selection (Cholesky) + cosine merge
            if self.voxel_size > 0:
                p2v_fix = voxelize(feat_xyz, self.voxel_size)
            else:
                N_tok = V * H * W
                p2v_fix = torch.arange(N_tok, device=video_features.device).unsqueeze(0).expand(B, -1)
            MERGE_CUTOFF = float(os.environ.get('MERGE_CUTOFF', '5.0'))
            target_k = int(self.adaptive_ratio) if self.adaptive_ratio >= 1 else (int(self.adaptive_ratio * V * H * W) if self.adaptive_ratio > 0 else 0)
            pooled_list, xyz_list, counts = [], [], []
            for b in range(B):
                vox_feat = scatter_mean(video_features[b], p2v_fix[b], dim=0)  # (n_vox, C)
                vox_xyz = scatter_mean(video_xyz[b], p2v_fix[b], dim=0)        # (n_vox, 3)
                n_vox = vox_feat.shape[0]
                k = min(target_k, n_vox) if target_k > 0 else n_vox

                if k < n_vox and k >= 2:
                    # --- Stage 1: DPP greedy MAP selection (Cholesky-based) ---
                    feat_norm = F.normalize(vox_feat.float(), dim=-1)
                    L = feat_norm @ feat_norm.T  # cosine similarity kernel
                    d = L.diag().clone()
                    C = torch.zeros(k, n_vox, device=vox_feat.device, dtype=torch.float32)
                    selected = []
                    for t in range(k):
                        j = d.argmax().item()
                        selected.append(j)
                        if t < k - 1:
                            c_j = L[j].clone()
                            if t > 0:
                                c_j -= C[:t, j] @ C[:t]
                            C[t] = c_j / torch.sqrt(d[j].clamp(min=1e-8))
                            d -= C[t] ** 2
                            d.clamp_(min=0)
                        d[j] = -float('inf')

                    sel_idx = torch.tensor(selected, device=vox_feat.device)

                    # --- Stage 2: Merge remaining voxels into nearest anchor ---
                    anchor_feat = vox_feat[sel_idx]
                    anchor_xyz = vox_xyz[sel_idx]
                    mask = torch.ones(n_vox, dtype=torch.bool, device=vox_feat.device)
                    mask[sel_idx] = False
                    remain_idx = torch.where(mask)[0]

                    if remain_idx.numel() > 0:
                        remain_feat = vox_feat[remain_idx]
                        remain_xyz = vox_xyz[remain_idx]
                        remain_norm = F.normalize(remain_feat.float(), dim=-1)
                        anchor_norm = F.normalize(anchor_feat.float(), dim=-1)
                        cos_dist = 1.0 - (remain_norm @ anchor_norm.T)
                        _, nearest_anchor = cos_dist.min(dim=1)

                        # Spatial cutoff: only merge if within MERGE_CUTOFF voxel grids
                        if self.voxel_size > 0:
                            anchor_grid = torch.floor(anchor_xyz / self.voxel_size)
                            remain_grid = torch.floor(remain_xyz / self.voxel_size)
                            xyz_grid_dist = torch.norm(remain_grid.float() - anchor_grid[nearest_anchor].float(), dim=-1)
                            close_mask = xyz_grid_dist <= MERGE_CUTOFF
                        else:
                            xyz_dist = torch.norm(remain_xyz.float() - anchor_xyz[nearest_anchor].float(), dim=-1)
                            close_mask = xyz_dist <= MERGE_CUTOFF

                        if close_mask.any():
                            close_anchor = nearest_anchor[close_mask]
                            close_feat = remain_feat[close_mask]
                            ones = torch.ones(close_anchor.shape[0], dtype=vox_feat.dtype, device=vox_feat.device)
                            merge_sum = scatter_add(close_feat, close_anchor, dim=0, dim_size=k)
                            merge_cnt = scatter_add(ones, close_anchor, dim=0, dim_size=k)
                            total_cnt = 1.0 + merge_cnt
                            merged_feat = (anchor_feat + merge_sum) / total_cnt.unsqueeze(-1)
                        else:
                            merged_feat = anchor_feat
                    else:
                        merged_feat = anchor_feat

                    pooled_list.append(merged_feat)
                    xyz_list.append(anchor_xyz)
                else:
                    pooled_list.append(vox_feat)
                    xyz_list.append(vox_xyz)
                counts.append(pooled_list[-1].shape[0])

            pooled_video_features = torch.cat(pooled_list, dim=0)
            batch_offset = torch.tensor(counts, dtype=torch.int32, device=video_features.device).cumsum(0)

        else:
            raise NotImplementedError(f"Unknown pooling: '{self.pooling}'")

        self._logged_once = True
        return pooled_video_features, batch_offset

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
