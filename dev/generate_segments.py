import json
import argparse
from pathlib import Path
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
from PIL import Image
import os
import cv2

def load_video(path, height=None, width=None):
    # load all frames from video
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if height is not None and width is not None:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()
    return frames

def main(args):
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                f"{args.model_path}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{args.model_path}/diffusion_pytorch_model-00002-of-00007.safetensors", 
                f"{args.model_path}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{args.model_path}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{args.model_path}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{args.model_path}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{args.model_path}/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            f"{args.model_path}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.model_path}/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_models(
        [f"{args.model_path}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float32,
    )

    # Initialize pipeline
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)

    # Read JSON file
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # Generate segments for each sample
    for sample in data:
        # Extract frames from video
        video_path = sample['video_path']
        cap = cv2.VideoCapture(video_path)
        
        # Get first frame (0th frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        first_frame = Image.fromarray(first_frame).resize((sample['width'], sample['height']))
        # first_frame = Image.fromarray(first_frame)
        
        # Get last frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        last_frame = Image.fromarray(last_frame).resize((sample['width'], sample['height']))
        # last_frame = Image.fromarray(last_frame)
        
        cap.release()

        for r in range(args.num_repeats):
            # Generate video
            video = pipe(
                prompt=sample['prompt'],
                negative_prompt=sample['negative_prompt'],
                num_inference_steps=sample['num_inference_steps'],
                input_image=first_frame,
                end_image=last_frame,
                height=sample['height'],
                width=sample['width'],
                seed=sample['seed']+r,
                tiled=sample['tiled'],
                num_frames=sample['num_frames'],
                input_video=load_video(sample['video_path'], sample['height'], sample['width'])
            )

            sample_path = Path(sample['video_path']).parent / args.version
            output_path = sample_path / f"segment_{sample['sample_id']}_{r}.mp4"
            
            # Save the generated video
            os.makedirs(sample_path, exist_ok=True)
            save_video(video, output_path, fps=args.fps, quality=args.quality)

        print(f"Generated segment {sample['sample_id']} saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate video segments using WanVideoPipeline')
    parser.add_argument('--model-path', type=str, default="models/Wan-AI/Wan2.1-FLF2V-14B-720P",
                        help='Path to model files')
    parser.add_argument('--json-path', type=str, required=True,
                        help='Path to JSON file containing sample data')
    parser.add_argument('--num-repeats', type=int, default=2,
                        help='Number of times to repeat generation for each sample')
    parser.add_argument('--fps', type=int, default=15,
                        help='FPS for output video')
    parser.add_argument('--quality', type=int, default=8,
                        help='Quality setting for output video (0-10)')
    parser.add_argument('--version', type=str, default="",
                        help='Version of the output video')
    
    args = parser.parse_args()
    main(args)
