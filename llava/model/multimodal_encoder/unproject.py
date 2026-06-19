import torch
from torch.nn import functional as F


def unproject(intrinsics, poses, depths):
    """
    Inputs:
        intrinsics: B X V X 3 X 3
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)
    
    Outputs:
        world_coords: B X V X H X W X 3
    """
    # (B, V, 336, 336)
    B, V, H, W = depths.shape 
    # (B, V, 1)
    fx, fy, px, py = intrinsics[..., 0, 0][..., None], intrinsics[..., 1, 1][..., None], intrinsics[..., 0, 2][..., None], intrinsics[..., 1, 2][..., None]

    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    z = depths.flatten(2)  # (B, V, H*W)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)

    return world_coords

def backproject_depth(depths, poses, intrinsics=None):
    B, V, H, W = depths.shape
    xyz = unproject(intrinsics, poses, depths)
    return xyz  # (B X V X H X W X 3)

def interpolate_depth(xyz, multi_scale_features, method="nearest"):
    multi_scale_xyz = []
    B, V, H, W, _ = xyz.shape
    for feat in multi_scale_features:
        h, w = feat.shape[2:]
        xyz_ = torch.nn.functional.interpolate(
            xyz.reshape(B*V, H, W, 3).permute(0, 3, 1, 2), size=(h, w),
            mode=method).permute(0, 2, 3, 1).reshape(B, V, h, w, 3)
        multi_scale_xyz.append(xyz_)    
    return multi_scale_xyz

def backprojector_dataloader(
    multi_scale_features, depths, poses,
    intrinsics=None, method='nearest', 
    padding=None):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 1024, 24, 24], [B*V, 1024, 48, 48], [B*V, 1024, 96, 96]
        depths: tensor [B, 5, 336, 336]
        poses: tensor [B, 5, 4, 4]
        intrinsics: tensor [B, 5, 4, 4]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    # (B, V, H, W, 3)
    new_xyz = backproject_depth(
        depths, poses, intrinsics)

    if padding is not None:
        new_xyz = F.pad(new_xyz.permute(0, 1, 4, 2, 3), (0, padding[1], 0, padding[0]), mode='constant', value=0).permute(0, 1, 3, 4, 2)

    multi_scale_xyz = interpolate_depth(
        new_xyz, multi_scale_features,
        method=method)

    if len(multi_scale_xyz) == 1:
        multi_scale_xyz =  multi_scale_xyz[0]
    
    return multi_scale_xyz, new_xyz

def voxelize(xyz, voxel_size=0.28):
    """
    Inputs:
        xyz: list of tensors [B, V, H, W, 3]
        voxel_size: voxel size

    Outputs:
        N=V*H*W
        p2v: tensors [B, N]
    """
    B, V, H, W, _ = xyz.shape
    xyz = xyz.reshape(B, V*H*W, 3)
    p2v = voxelization(xyz, voxel_size)
    return p2v


def adaptive_voxelize(xyz, target_tokens=576, v_min=0.05, v_max=2.0, max_iter=20):
    """
    Binary search for voxel_size that produces ~target_tokens voxels.

    Inputs:
        xyz: tensor [B, V, H, W, 3]
        target_tokens: desired number of voxels
        v_min: minimum voxel size (meters)
        v_max: maximum voxel size (meters)
        max_iter: binary search iterations

    Outputs:
        p2v: tensor [B, N], point-to-voxel mapping
        voxel_size: float, the found voxel size
    """
    B, V, H, W, _ = xyz.shape
    xyz_flat = xyz.reshape(B, V * H * W, 3)

    lo, hi = v_min, v_max
    best_p2v = None
    best_vs = (lo + hi) / 2

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p2v = voxelization(xyz_flat, mid)
        n_voxels = (p2v.max(1)[0] + 1).float().mean().item()

        best_p2v = p2v
        best_vs = mid

        if abs(n_voxels - target_tokens) / max(target_tokens, 1) < 0.05:
            break
        if n_voxels > target_tokens:
            lo = mid
        else:
            hi = mid

    return best_p2v, best_vs


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0].to(torch.long)
    arr_max = arr.max(1, keepdims=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long).to(arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys

def voxelization(xyz, voxel_size):
    """
    Inputs:
        xyz: tensor [B, N, 3]
        voxel_size: float
    Outputs: 
        point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
    """
    B, N, _ = xyz.shape
    xyz = xyz / voxel_size
    xyz = torch.round(xyz).long()
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)

    point_to_voxel = torch.stack(
        [torch.unique(keys[b], return_inverse=True)[1] for b in range(B)], 0)
    return point_to_voxel


def spatial_token_merging(features, feat_xyz, v_init=0.05, delta_v=0.05,
                          r_percent=70, target_tokens=928, max_iter=20):
    """
    Spatial Token Merging (SToMe)

    ToMe-style bipartite matching constrained to 3D voxels.
    Instead of dropping tokens (DTC), merges matched pairs via
    size-weighted averaging so information is preserved.

    Args:
        features: (B, N, C) visual token features
        feat_xyz: (B, V, H, W, 3) 3D world coordinates per patch
        v_init: initial voxel size
        delta_v: voxel size increment per iteration
        r_percent: percentage of most similar edges to keep (0-100)
        target_tokens: desired token count per batch element
        max_iter: safety cap on iterations

    Returns:
        compressed_features: (sum_of_tokens, C) concatenated across batch
        batch_offset: (B,) int32 cumulative token counts
    """
    B, V, H, W, _ = feat_xyz.shape
    N = V * H * W
    xyz_flat = feat_xyz.reshape(B, N, 3)
    device = features.device
    dtype = features.dtype
    C = features.shape[-1]

    all_compressed = []
    counts = []

    for b in range(B):
        feat = features[b].clone()                        # (N, C)
        pos = xyz_flat[b].clone().to(dtype)               # (N, 3)
        size = torch.ones(N, device=device, dtype=dtype)  # tracks merged token count
        alive = torch.arange(N, device=device)

        voxel_size = v_init

        for _it in range(max_iter):
            M = len(alive)
            if M <= target_tokens:
                break

            cur_feat = feat[alive]                        # (M, C)
            cur_pos = pos[alive].unsqueeze(0)             # (1, M, 3)

            # --- voxelize alive tokens ---
            p2v = voxelization(cur_pos, voxel_size)[0]    # (M,)

            # --- random partition into src / dst ---
            is_src = torch.rand(M, device=device) < 0.5
            if is_src.sum() == 0 or (~is_src).sum() == 0:
                voxel_size += delta_v
                continue

            src_idx = torch.where(is_src)[0]
            dst_idx = torch.where(~is_src)[0]

            # --- cosine similarity (|src| x |dst|) ---
            feat_norm = F.normalize(cur_feat, dim=-1)
            sim = feat_norm[src_idx] @ feat_norm[dst_idx].T

            # mask: same voxel only
            same_voxel = (p2v[src_idx].unsqueeze(1) == p2v[dst_idx].unsqueeze(0))
            sim[~same_voxel] = -float('inf')

            max_sim, max_dst_local = sim.max(dim=1)
            valid = max_sim > -float('inf')

            if valid.sum() == 0:
                voxel_size += delta_v
                continue

            valid_src = src_idx[valid]
            valid_dst = dst_idx[max_dst_local[valid]]
            valid_sim = max_sim[valid]

            # --- keep top r% most similar edges ---
            n_edges = len(valid_sim)
            n_keep = max(1, int(n_edges * r_percent / 100))
            _, top_k = valid_sim.topk(min(n_keep, n_edges))

            merge_src_local = valid_src[top_k]
            merge_dst_local = valid_dst[top_k]

            src_orig = alive[merge_src_local]
            dst_orig = alive[merge_dst_local]

            # --- merge: size-weighted average (vectorized via scatter) ---
            src_w = size[src_orig]                            # (K,)

            feat_delta = torch.zeros(N, C, device=device, dtype=feat.dtype)
            feat_delta.scatter_add_(
                0, dst_orig.unsqueeze(1).expand(-1, C),
                feat[src_orig] * src_w.unsqueeze(1))

            pos_delta = torch.zeros(N, 3, device=device, dtype=pos.dtype)
            pos_delta.scatter_add_(
                0, dst_orig.unsqueeze(1).expand(-1, 3),
                pos[src_orig] * src_w.unsqueeze(1))

            size_delta = torch.zeros(N, device=device, dtype=dtype)
            size_delta.scatter_add_(0, dst_orig, src_w)

            updated = size_delta > 0
            new_total = size[updated] + size_delta[updated]
            feat[updated] = (feat[updated] * size[updated].unsqueeze(1)
                             + feat_delta[updated]) / new_total.unsqueeze(1)
            pos[updated] = (pos[updated] * size[updated].unsqueeze(1)
                            + pos_delta[updated]) / new_total.unsqueeze(1)
            size[updated] = new_total

            # remove merged src tokens
            keep_mask = torch.ones(M, dtype=torch.bool, device=device)
            keep_mask[merge_src_local] = False
            alive = alive[keep_mask]

            voxel_size += delta_v

        all_compressed.append(feat[alive])
        counts.append(len(alive))

    compressed_features = torch.cat(all_compressed, dim=0)
    batch_offset = torch.tensor(counts, dtype=torch.int32,
                                device=device).cumsum(0)
    return compressed_features, batch_offset


def dynamic_token_compression(features, feat_xyz, v_init=0.1, delta_v=0.1,
                                r_percent=50, target_tokens=576, max_iter=100):
    """
    Dynamic Token Compression (DTC) - CVPR 2025
    Huang et al., "Zero-shot 3D QA via Voxel-based Dynamic Token Compression"

    Iteratively compresses visual tokens via bipartite soft matching
    within dynamically-sized voxels. Tokens from distinct objects are
    preserved by gating merges on visual similarity.

    Args:
        features: (B, N, C) visual token features
        feat_xyz: (B, V, H, W, 3) 3D world coordinates per patch
        v_init: initial voxel size
        delta_v: voxel size increment per iteration
        r_percent: percentage of most similar edges to keep (0-100)
        target_tokens: desired token count per batch element
        max_iter: safety cap on iterations

    Returns:
        compressed_features: (sum_of_tokens, C) concatenated across batch
        batch_offset: (B,) int32 cumulative token counts
        all_alive: list of (k,) index tensors per batch element
    """
    B, V, H, W, _ = feat_xyz.shape
    N = V * H * W
    xyz_flat = feat_xyz.reshape(B, N, 3)
    device = features.device

    all_compressed = []
    all_alive = []
    counts = []

    for b in range(B):
        feat = features[b]       # (N, C)
        pos = xyz_flat[b]        # (N, 3)
        alive = torch.arange(N, device=device)

        voxel_size = v_init
        stall_count = 0  # track consecutive no-progress iterations

        for _it in range(max_iter):
            M = len(alive)
            if M <= target_tokens:
                break

            cur_feat = feat[alive]                        # (M, C)
            cur_pos = pos[alive].unsqueeze(0)             # (1, M, 3)

            # --- voxelize alive tokens ---
            p2v = voxelization(cur_pos, voxel_size)[0]    # (M,)

            # --- random partition into A / B ---
            is_A = torch.rand(M, device=device) < 0.5
            if is_A.sum() == 0 or (~is_A).sum() == 0:
                voxel_size += delta_v
                continue

            A_idx = torch.where(is_A)[0]    # local indices
            B_idx = torch.where(~is_A)[0]

            # --- cosine similarity (|A| x |B|) ---
            feat_norm = F.normalize(cur_feat, dim=-1)
            sim = feat_norm[A_idx] @ feat_norm[B_idx].T   # (|A|, |B|)

            # mask: allow matching only within the same voxel
            # if stalled too long, drop voxel constraint to force progress
            if stall_count < 3:
                same_voxel = (p2v[A_idx].unsqueeze(1) == p2v[B_idx].unsqueeze(0))
                sim[~same_voxel] = -float('inf')

            # for each A token, find most similar B token in same voxel
            max_sim, max_b_local = sim.max(dim=1)         # (|A|,)
            valid = max_sim > -float('inf')

            if valid.sum() == 0:
                stall_count += 1
                voxel_size += delta_v
                continue

            valid_A   = A_idx[valid]
            valid_B   = B_idx[max_b_local[valid]]
            valid_sim = max_sim[valid]

            # --- keep top r% most similar edges (global ranking) ---
            # cap: never remove more than (M - target_tokens) to hit target precisely
            n_edges = len(valid_sim)
            n_keep  = max(1, int(n_edges * r_percent / 100))
            max_remove = M - target_tokens
            if max_remove > 0:
                n_keep = min(n_keep, max_remove)
            _, top_k = valid_sim.topk(min(n_keep, n_edges))

            kept_A = valid_A[top_k]
            kept_B = valid_B[top_k]

            # for each kept edge, remove the token with larger original
            # index (= later in 2D image order)
            orig_A = alive[kept_A]
            orig_B = alive[kept_B]
            remove_local = torch.where(orig_A > orig_B, kept_A, kept_B)

            keep_mask = torch.ones(M, dtype=torch.bool, device=device)
            keep_mask[remove_local] = False

            prev_M = M
            alive = alive[keep_mask]
            # reset stall if progress was made
            if len(alive) < prev_M:
                stall_count = 0
            else:
                stall_count += 1
            voxel_size += delta_v

        all_compressed.append(feat[alive])
        all_alive.append(alive)
        counts.append(len(alive))

    compressed_features = torch.cat(all_compressed, dim=0)
    batch_offset = torch.tensor(counts, dtype=torch.int32,
                                device=device).cumsum(0)
    return compressed_features, batch_offset, all_alive


def voxel_map_to_source(voxel_map, poin2voxel):
    """
    Input:
        voxel_map (B, N1, C)
        point2voxel (B, N)
    Output:
        src_new (B, N, C)
    """
    bs, n, c = voxel_map.shape
    src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
    return src_new