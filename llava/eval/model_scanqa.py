import argparse
import torch
import os
import json
import time
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    n_visual_list = []
    latency_list = []
    memory_list = []
    pbar = tqdm(questions)
    for line in pbar:
        idx = line["question_id"]
        video_file = line["video"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        videos_dict = process_videos(
            video_path,
            processor['video'],
            mode='uniform',
            device=model.device,
            text=cur_prompt
        )

        images_tensor = videos_dict['images'].to(model.device, dtype=torch.bfloat16)
        depths_tensor = videos_dict['depths'].to(model.device, dtype=torch.bfloat16)
        poses_tensor = videos_dict['poses'].to(model.device, dtype=torch.bfloat16)
        intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch.bfloat16)


        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                intrinsics=intrinsics_tensor,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=512,
                use_cache=True,
            )
        torch.cuda.synchronize()
        latency_list.append(time.time() - t_start)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Track visual token count + KV cache calculation
        video_tower = model.get_model().get_video_tower()
        if hasattr(video_tower, '_voxel_xyz') and video_tower._voxel_xyz is not None:
            _n_vis = video_tower._voxel_xyz[0].shape[0]
            n_visual_list.append(_n_vis)
            # KV cache size: 2(K,V) × n_layers × seq_len × n_heads × head_dim × 2(bf16)
            cfg = model.config
            n_layers = cfg.num_hidden_layers
            n_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
            head_dim = cfg.hidden_size // cfg.num_attention_heads
            seq_len = input_ids.shape[1] - 1 + _n_vis  # text tokens + visual tokens
            kv_bytes = 2 * n_layers * seq_len * n_kv_heads * head_dim * 2  # bf16=2bytes
            kv_mb = kv_bytes / 1024**2
            memory_list.append(kv_mb)
            postfix = dict(v_num=_n_vis, ms=f"{latency_list[-1]*1000:.0f}", kv=f"{kv_mb:.0f}MB")
            if hasattr(video_tower, '_sel_times') and video_tower._sel_times:
                postfix["sel_ms"] = f"{video_tower._sel_times[-1]:.1f}"
            if hasattr(video_tower, '_algo_times') and video_tower._algo_times:
                postfix["algo_ms"] = f"{video_tower._algo_times[-1]:.1f}"
            pbar.set_postfix(**postfix)

        # === LLM attention -> voxel visualization (--visualize flag) ===
        if args.visualize:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            vis_dir = os.path.join(os.path.dirname(answers_file), 'llm_attn_vis')
            os.makedirs(vis_dir, exist_ok=True)

            with torch.inference_mode():
                fwd_out = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    depths=depths_tensor,
                    poses=poses_tensor,
                    intrinsics=intrinsics_tensor,
                    output_attentions=True,
                    return_dict=True,
                )
            # voxel xyz
            video_tower = model.get_model().get_video_tower()
            voxel_xyz = video_tower._voxel_xyz[0].cpu().float()  # (n_vox, 3)
            n_visual = voxel_xyz.shape[0]

            # Locate visual token positions
            image_pos = (input_ids[0] == -200).nonzero(as_tuple=True)[0][0].item()
            vis_start = image_pos
            vis_end = vis_start + n_visual

            # Mid-layer (16th) attention: (B, n_heads, seq_len, seq_len)
            last_attn = fwd_out.attentions[15].float()
            last_attn = last_attn.mean(dim=1)  # avg over heads: (B, seq_len, seq_len)
            # query (text after visual) -> visual token attention
            query_to_vis = last_attn[0, vis_end:, vis_start:vis_end]  # (n_query, n_visual)
            avg_attn = query_to_vis.mean(dim=0).cpu().numpy()  # (n_visual,)

            # === Attention distribution analysis: sum of attention from query tokens to each region ===
            seq_len = last_attn.shape[-1]
            # query tokens (vis_end ~ end) -> attention to each region
            query_attn = last_attn[0, vis_end:, :]  # (n_query, seq_len)
            attn_to_system = query_attn[:, :vis_start].sum(dim=-1).mean().item()       # -> system
            attn_to_visual = query_attn[:, vis_start:vis_end].sum(dim=-1).mean().item() # -> 3D visual
            attn_to_query  = query_attn[:, vis_end:].sum(dim=-1).mean().item()          # -> query text
            print(f"[attn dist] {idx} | system={attn_to_system:.4f}, 3D_visual={attn_to_visual:.4f}, "
                  f"query_text={attn_to_query:.4f} | "
                  f"n_sys={vis_start}, n_vis={n_visual}, n_query={seq_len - vis_end}, total={seq_len}")

            xyz = voxel_xyz.numpy()

            # === Load RGB point cloud ===
            from PIL import Image as PILImage
            scene_dir = video_path if os.path.isdir(video_path) else os.path.join(args.video_folder, video_file)
            color_dir = os.path.join(scene_dir, 'color')
            depth_dir = os.path.join(scene_dir, 'depth')
            pose_dir_path = os.path.join(scene_dir, 'pose')
            intrinsic_file = os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt')
            K = np.loadtxt(intrinsic_file)[:3, :3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            axis_file = os.path.join(scene_dir, 'axis_align_matrix.txt')
            axis_align = np.loadtxt(axis_file).reshape(4, 4) if os.path.exists(axis_file) else np.eye(4)

            rgb_frames = sorted([int(f.split('.')[0]) for f in os.listdir(depth_dir) if f.endswith('.png')])
            step = max(1, len(rgb_frames) // 15)
            rgb_frames = rgb_frames[::step][:15]
            all_pts, all_rgb = [], []
            for fi in rgb_frames:
                pose_f = np.loadtxt(os.path.join(pose_dir_path, f'{fi}.txt'))
                if np.any(np.isinf(pose_f)) or np.any(np.isnan(pose_f)):
                    continue
                dep = np.array(PILImage.open(os.path.join(depth_dir, f'{fi}.png')), dtype=np.float32) / 1000.0
                Hd, Wd = dep.shape
                col_path = os.path.join(color_dir, f'{fi}.jpg')
                if not os.path.exists(col_path):
                    col_path = os.path.join(color_dir, f'{fi}.png')
                col = np.array(PILImage.open(col_path).convert('RGB').resize((Wd, Hd))) / 255.0
                v_idx, u_idx = np.where(dep > 0.1)
                z_val = dep[v_idx, u_idx]
                x_val = (u_idx - cx) * z_val / fx
                y_val = (v_idx - cy) * z_val / fy
                pts_cam = np.stack([x_val, y_val, z_val, np.ones_like(z_val)], axis=-1)
                pts_w = (axis_align @ pose_f @ pts_cam.T).T[:, :3]
                if len(pts_w) > 3000:
                    si = np.random.choice(len(pts_w), 3000, replace=False)
                    pts_w = pts_w[si]; rgb_vals = col[v_idx[si], u_idx[si]]
                else:
                    rgb_vals = col[v_idx, u_idx]
                all_pts.append(pts_w); all_rgb.append(rgb_vals)
            scene_pts = np.concatenate(all_pts, axis=0)
            scene_rgb = np.concatenate(all_rgb, axis=0)

            fig = plt.figure(figsize=(28, 8))
            fig.suptitle(f'[{idx}] Q: {cur_prompt}\nA: {outputs}', fontsize=10, wrap=True)

            # Left: attention by voxel index
            ax1 = fig.add_subplot(131)
            ax1.bar(np.arange(n_visual), avg_attn, color='coral', alpha=0.7, width=1.0)
            ax1.set_xlabel('Voxel Index')
            ax1.set_ylabel('LLM Attention')
            ax1.set_title(f'Attention by Index (n={n_visual})')
            top_k = min(10, n_visual)
            top_idx = np.argsort(avg_attn)[-top_k:]
            for ti in top_idx:
                ax1.annotate(f'{ti}', (ti, avg_attn[ti]), fontsize=6, ha='center', va='bottom')

            # Middle: RGB point cloud
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                        c=scene_rgb, s=0.3, alpha=0.5)
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            ax2.set_title(f'RGB Point Cloud ({len(scene_pts)} pts)')

            # Right: 3D attention heatmap
            ax3 = fig.add_subplot(133, projection='3d')
            sc = ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                             c=avg_attn, cmap='hot', s=15, alpha=0.8)
            plt.colorbar(sc, ax=ax3, label='LLM Attention', shrink=0.6)
            ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
            ax3.set_title('3D Attention Heatmap')

            vis_prefix = 'scanqa'
            save_path = os.path.join(vis_dir, f'{vis_prefix}_{idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            # === Plotly interactive HTML ===
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError:
                print(f"[VIS] {save_path}  (plotly not installed, skipping HTML)")
                del fwd_out, last_attn
                # skip to answer writing
                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({"question_id": idx,
                                           "prompt": cur_prompt,
                                           "text": outputs,
                                           "answer_id": ans_id,
                                           "model_id": model_name,
                                           "metadata": {}}) + "\n")
                ans_file.flush()
                continue

            fig_html = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=['RGB Point Cloud', '3D Attention Heatmap'],
                horizontal_spacing=0.02,
            )

            # RGB point cloud
            rgb_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in scene_rgb]
            fig_html.add_trace(go.Scatter3d(
                x=scene_pts[:, 0], y=scene_pts[:, 1], z=scene_pts[:, 2],
                mode='markers',
                marker=dict(size=1.2, color=rgb_str, opacity=0.7),
                name='RGB',
            ), row=1, col=1)

            # Attention heatmap on voxels
            fig_html.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=avg_attn,
                    colorscale='Hot',
                    opacity=0.85,
                    colorbar=dict(title='Attn', x=1.0, len=0.6),
                    cmin=0,
                ),
                text=[f'vox {i}<br>attn={avg_attn[i]:.6f}' for i in range(n_visual)],
                hoverinfo='text',
                name='Attention',
            ), row=1, col=2)

            scene_cfg = dict(aspectmode='data')
            fig_html.update_layout(
                title=f'[{idx}] Q: {cur_prompt}<br>A: {outputs}',
                width=1600, height=700,
                scene=scene_cfg, scene2=scene_cfg,
                showlegend=False,
            )

            html_path = os.path.join(vis_dir, f'{vis_prefix}_{idx}.html')
            fig_html.write_html(html_path)
            print(f"[VIS] {save_path}  [HTML] {html_path}  (n_vox={n_visual}, n_query={query_to_vis.shape[0]})")
            del fwd_out, last_attn

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

    # Print visual token statistics
    if n_visual_list:
        import numpy as np
        arr = np.array(n_visual_list)
        print(f"\n[Visual Tokens] avg={arr.mean():.1f}, min={arr.min()}, max={arr.max()}, std={arr.std():.1f}, n={len(arr)}")

    # Print latency statistics
    if latency_list:
        import numpy as np
        lat = np.array(latency_list)
        print(f"[Latency] avg={lat.mean()*1000:.1f}ms, min={lat.min()*1000:.1f}ms, max={lat.max()*1000:.1f}ms, std={lat.std()*1000:.1f}ms, n={len(lat)}")

    # Print KV cache statistics
    if memory_list:
        import numpy as np
        mem = np.array(memory_list)
        print(f"[KV Cache] avg={mem.mean():.0f}MB, min={mem.min():.0f}MB, max={mem.max():.0f}MB, n={len(mem)}")

    # Pooling algorithm timing statistics
    video_tower = model.get_model().get_video_tower()
    if hasattr(video_tower, '_pool_times') and video_tower._pool_times:
        import numpy as np
        pt = np.array(video_tower._pool_times)
        print(f"[Pooling] avg={pt.mean():.1f}ms, min={pt.min():.1f}ms, max={pt.max():.1f}ms, std={pt.std():.1f}ms, n={len(pt)}")
    if hasattr(video_tower, '_sel_times') and video_tower._sel_times:
        import numpy as np
        st = np.array(video_tower._sel_times)
        print(f"[Selection] avg={st.mean():.1f}ms, min={st.min():.1f}ms, max={st.max():.1f}ms, std={st.std():.1f}ms, n={len(st)}")
    if hasattr(video_tower, '_algo_times') and video_tower._algo_times:
        import numpy as np
        at = np.array(video_tower._algo_times)
        print(f"[Algorithm] avg={at.mean():.1f}ms, min={at.min():.1f}ms, max={at.max():.1f}ms, std={at.std():.1f}ms, n={len(at)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava3d-v1.5-7b-task-v3-tuning")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="playground/data/LLaVA-3D-Pretrain")
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_sqa3d_val_question.json")
    parser.add_argument("--answers-file", type=str, default="./llava3d_sqa3d_val_answer_pred.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", help="LLM attention voxel visualization")
    args = parser.parse_args()

    eval_model(args)
