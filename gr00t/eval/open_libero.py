# source /pi/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/activate
import argparse
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import image_tools

# ================= 配置区域 =================
# 对应你的 tree 结构
BASE_DIR = "/pi/open_loop_data/libero"
DEFAULT_PARQUET = f"{BASE_DIR}/episode_000000.parquet"
DEFAULT_VIDEO_WRIST = f"{BASE_DIR}/wrist/episode_000000_h264.mp4"
DEFAULT_VIDEO_LEFT  = f"{BASE_DIR}/left/episode_000000.mp4"
DEFAULT_VIDEO_RIGHT = f"{BASE_DIR}/right/episode_000000.mp4"
DEFAULT_VIDEO_IMAGE = f"{BASE_DIR}/image/episode_000000_h264.mp4"

# 图像处理参数
RESIZE_SIZE = 224
SERVER_PORT = 8001
# ===========================================

def process_frame(frame_bgr):
    """
    处理单帧: BGR -> RGB -> Flip(视情况) -> Resize -> CHW
    """
    if frame_bgr is None:
        raise ValueError("读到了空帧")

    # 1. OpenCV 读取的是 BGR，转 RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 2. 翻转处理
    # 注意：Robocasa/Mujoco 原始渲染通常是倒置的。
    # 如果你的 mp4 已经是正的（比如经过了转码），请注释掉下面这行！
    # img_rgb = img_rgb[::-1, ::-1]

    # 3. Resize & Pad
    img_batched = img_rgb[np.newaxis, ...]
    resized = image_tools.resize_with_pad(img_batched, RESIZE_SIZE, RESIZE_SIZE)
    
    # 4. (H, W, C) -> (C, H, W)
    return image_tools.convert_to_uint8(resized[0]).transpose(2, 0, 1)

def plot_results(gt_action, pred_action, save_path):
    dims = gt_action.shape[1]
    
    # 动态生成 dim0, dim1, ... dimN
    dim_names = [f"dim{i}" for i in range(dims)]
    
    fig, axes = plt.subplots(nrows=dims, ncols=1, figsize=(10, 2 * dims), sharex=True)
    if dims == 1: axes = [axes]

    fig.suptitle(f"Open Loop Replay: GT (Blue) vs Pred (Red)", fontsize=16)

    total_mse = 0
    for i in range(dims):
        ax = axes[i]
        ax.plot(gt_action[:, i], label="GT", color='blue', linewidth=2, alpha=0.6)
        ax.plot(pred_action[:, i], label="Pred", color='red', linestyle='--', linewidth=1.5)
        
        mse = np.mean((gt_action[:, i] - pred_action[:, i]) ** 2)
        total_mse += mse
        
        name = dim_names[i]
        ax.set_title(f"{name} (MSE: {mse:.5f})")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 结果图已保存至: {save_path}")
    print(f"📉 Average MSE (Dims 5-12): {total_mse / dims:.6f}")

def run_eval(args):
    # 1. 连接 Server
    print(f"🔌 连接 Server: {args.host}:{args.port}...")
    try:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 2. 加载 Parquet
    print(f"📂 加载 Parquet: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    
    # 3. 初始化视频读取器
    # video_map = {
    #     "wrist": args.video_wrist,
    #     "left":  args.video_left,
    #     "right": args.video_right
    # }
    video_map = {
        "wrist": DEFAULT_VIDEO_WRIST,
        "image": DEFAULT_VIDEO_IMAGE,
    }
    caps = {}
    print(f"🎥 打开视频流:")
    for name, path in video_map.items():
        if Path(path).exists():
            caps[name] = cv2.VideoCapture(path)
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ 文件不存在: {path}")
            return

    # 确定总帧数
    min_video_frames = min([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps.values()])
    total_steps = min(len(df), min_video_frames, args.steps)
    print(f"⏱️ 计划评估帧数: {total_steps}")
    print(f"📝 任务指令: \"{args.task_text}\"")

    pred_list, gt_list = [], []

    # 4. 循环推理
    for i in tqdm(range(total_steps), desc="Replaying"):
        row = df.iloc[i]
        
        # --- A. 读取图像 ---
        imgs = {}
        for name, cap in caps.items():
            ret, frame = cap.read()
            if not ret: break
            imgs[name] = process_frame(frame)
        
        try:
            # --- B. 构造 Request ---
            # state_vec = np.array(row['observation.state']).astype(np.float32)
            
            # element = {
            #     "cam_high":        imgs["wrist"],
            #     "cam_left_wrist":  imgs["left"],
            #     "cam_right_wrist": imgs["right"],
            #     "state":           state_vec,
            #     "prompt":          args.task_text
            # }

            # element = {
            #     "images": {
            #         "cam_high": imgs["wrist"],
            #         "cam_left_wrist": imgs["left"],
            #         "cam_right_wrist": imgs["right"]
            #     },
            #     "state": state_vec,
            #     "prompt": args.task_text
            # }

            element = {
                "observation/image": imgs["image"],
                "observation/wrist_image": imgs["wrist"],
                "observation/state": np.array(row['state']).astype(np.float32),
                "prompt": args.task_text,
            }

            # --- C. 推理 ---
            output = client.infer(element)
            
            # --- D. 记录结果 ---
            # 取 Chunk 的第一帧
            pred_action = output["actions"][0]
            # gt_action   = np.array(row['action']).astype(np.float32)
            gt_action   = np.array(row['actions']).astype(np.float32)


            pred_list.append(pred_action)
            gt_list.append(gt_action)

        except Exception as e:
            print(f"❌ Error at step {i}: {e}")
            break

    # 5. 清理与画图
    for cap in caps.values(): cap.release()

    if len(pred_list) > 0:
        preds = np.array(pred_list)
        gts = np.array(gt_list)
        
        # print(f"✂️  正在截取 actions[5:12] 进行比较...")
        # preds_sliced = preds[:, 5:12]
        # gts_sliced   = gts[:, 5:12]

        save_path = f"/pi/replay_dim5-12.png"
        # plot_results(gts_sliced, preds_sliced, save_path)
        plot_results(gts, preds, save_path)
    else:
        print("❌ 未生成有效数据")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 路径参数 (默认值适配你的目录结构)
    parser.add_argument("--parquet_path", type=str, default=DEFAULT_PARQUET)
    parser.add_argument("--video_wrist",  type=str, default=DEFAULT_VIDEO_WRIST)
    parser.add_argument("--video_left",   type=str, default=DEFAULT_VIDEO_LEFT)
    parser.add_argument("--video_right",  type=str, default=DEFAULT_VIDEO_RIGHT)
    
    # Server 参数
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    
    # 任务参数
    # parser.add_argument("--task_text", type=str, default="close the left drawer")
    parser.add_argument("--task_text", type=str, default="put the white mug on the left plate and put the yellow and white mug on the right plate")
    parser.add_argument("--steps", type=int, default=500)

    args = parser.parse_args()
    run_eval(args)