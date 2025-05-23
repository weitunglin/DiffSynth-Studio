import json
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffsynth.data.video import save_video

def process_sample(sample, output_path):
    frame_path = sample / 'images'
    pose_path = sample / 'densepose'

    # get all frames
    frames = [f for f in frame_path.glob('*.png') if not f.name.startswith('.')]
    frames.sort()

    # get all masks  
    poses = [f for f in pose_path.glob('*.png') if not f.name.startswith('.')]
    poses.sort()

    assert len(frames) == len(poses)
    # Sample start frame i and number of 4-frame segments k
    # Total frames will be i + k*4 + 1
    min_k = 7
    max_k = 20
    
    # First randomly choose start frame i
    max_i = len(frames) - min_k*4 - 1
    if max_i < 0:
        raise ValueError(f"Video too short ({len(frames)} frames) for minimum k={min_k}")
    i = random.randint(0, max_i)
    
    # Then calculate max possible k given chosen i
    max_possible_k = (len(frames) - i - 1) // 4
    max_k = min(max_k, max_possible_k)
    
    if max_k < min_k:
        raise ValueError(f"Video too short ({len(frames)} frames) for minimum k={min_k} starting at frame {i}")
        
    # Sample k between min_k and max_k
    k = random.randint(min_k, max_k)
    
    j = i + k*4 + 1

    # sample frames
    segment_frames = frames[i:j]
    print(f'{len(frames)=}, {len(poses)=}, i={i}, k={k}, j={j}, len(segment_frames)={len(segment_frames)}')
    for i in range(len(segment_frames)):
        segment_frames[i] = Image.open(segment_frames[i])
    save_video(segment_frames, f'{output_path}/{sample.name}.mp4', fps=15, quality=8)

    return dict(
        sample_id=sample.name,
        video_path=f'{output_path}/{sample.name}.mp4',
        input_frame=i,
        end_frame=j,
        prompt='placeholder',
        negative_prompt='placeholder',
        seed=random.randint(0, 1000000),
        height=960,
        width=960,
        num_inference_steps=30,
        tiled=True,
        num_frames=len(segment_frames),
    )

def main(args):
    # get all samples
    all_samples = []
    for path in Path(args.base_path).iterdir():
        if path.is_dir():
            all_samples.append(path)

    print(f"Found {len(all_samples)} total samples")

    # sample num_samples samples
    random.seed(args.seed)
    random.shuffle(all_samples)

    samples = all_samples[:args.num_samples]

    all_sample_infos = []
    for sample in tqdm(samples, desc="Processing samples"):
        print(f"Processing {sample}")
        sample_info = process_sample(sample, args.output_path)
        all_sample_infos.append(sample_info)

    # save all samples to json
    output_json = f'{args.output_path}/devset_{args.num_samples}_samples.json'
    with open(output_json, 'w') as f:
        json.dump(all_sample_infos, f, indent=2)
    print(f"Saved sample info to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video samples for FLF2V')
    parser.add_argument('--base_path', type=str, default='/mnt/2sata/allen/data/tiktok-dataset/devset',
                        help='Base path containing the video samples')
    parser.add_argument('--output_path', type=str, default='/mnt/2sata/allen/data/tiktok-dataset/flf2v',
                        help='Output path for processed videos')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    main(args)
