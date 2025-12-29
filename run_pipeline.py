#!/usr/bin/env python3
"""
Automated 3D Gaussian Splatting Pipeline - Part 1: Video Processing
This script handles video-to-dataset conversion using FFmpeg and COLMAP.
"""
import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path


def run_ffmpeg_extract_frames(video_path, output_dir, max_frames=100, min_frames=30):
    """Extract frames from video using FFmpeg with dynamic frame count control."""
    print(f"[INFO] Extracting frames from {video_path}")

    # Get video duration first
    cmd_duration = [
        'ffprobe',
        '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        video_path
    ]

    try:
        result = subprocess.run(cmd_duration, capture_output=True, text=True, check=True)
        duration_seconds = float(result.stdout.strip())
        print(f"[INFO] Video duration: {duration_seconds:.2f} seconds")
    except (subprocess.CalledProcessError, ValueError):
        print("[WARNING] Could not get video duration, using default sampling")
        duration_seconds = 0

    # Calculate intelligent frame extraction strategy
    if duration_seconds > 0:
        # Calculate minimum required fps to meet min_frames
        min_required_fps = min_frames / duration_seconds

        # If video is short, sample at higher rate; if long, sample at lower rate
        if duration_seconds <= 30:
            # Short videos: sample at ~2 fps, but at least min_required_fps
            target_fps = max(2.0, min_required_fps)
        elif duration_seconds <= 120:
            # Medium videos: sample at ~1 fps, but at least min_required_fps
            target_fps = max(1.0, min_required_fps)
        else:
            # Long videos: sample at ~0.5 fps but ensure within max_frames
            target_fps = max(0.5, min_required_fps)
            target_fps = min(target_fps, max_frames / duration_seconds)

        # Calculate actual frame count
        actual_max_frames = int(duration_seconds * target_fps)
        actual_max_frames = max(actual_max_frames, min_frames)  # Ensure minimum
        actual_max_frames = min(actual_max_frames, max_frames)  # Ensure maximum
    else:
        # Fallback to fixed 1 fps if duration unknown
        target_fps = 1.0
        actual_max_frames = max(min_frames, min(max_frames, 100))  # Ensure within bounds

    print(f"[INFO] Sampling at {target_fps:.2f} fps (targeting {actual_max_frames} frames, min {min_frames}, max {max_frames})")

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={target_fps}',  # Dynamic frame rate
        '-q:v', '2',     # High quality
        '-frames:v', str(actual_max_frames),  # Limit total frames
        os.path.join(output_dir, 'frame_%06d.jpg')
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[SUCCESS] Extracted frames to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg failed: {e}")
        print(f"STDERR: {e.stderr}")
        return False


def run_colmap_sfm(images_dir, output_dir):
    """Run COLMAP SfM pipeline: feature_extractor -> feature_matcher -> mapper."""
    print(f"[INFO] Running COLMAP SfM on {images_dir}")

    # Create colmap database
    database_path = os.path.join(output_dir, 'database.db')
    cmd_db = [
        'colmap',
        'database_creator',
        '--database_path', database_path
    ]

    try:
        subprocess.run(cmd_db, check=True)
        print("[SUCCESS] Created COLMAP database")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create database: {e}")
        return False

    # Feature extraction - use OPENCV camera model for better distortion handling
    cmd_features = [
        'colmap',
        'feature_extractor',
        '--database_path', database_path,
        '--image_path', images_dir,
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'OPENCV'
    ]

    try:
        subprocess.run(cmd_features, check=True)
        print("[SUCCESS] Extracted features")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return False

    # Feature matching
    cmd_matcher = [
        'colmap',
        'exhaustive_matcher',
        '--database_path', database_path
    ]

    try:
        subprocess.run(cmd_matcher, check=True)
        print("[SUCCESS] Matched features")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Feature matching failed: {e}")
        return False

    # Sparse mapping
    sparse_dir = os.path.join(output_dir, 'sparse')
    os.makedirs(sparse_dir, exist_ok=True)

    cmd_mapper = [
        'colmap',
        'mapper',
        '--database_path', database_path,
        '--image_path', images_dir,
        '--output_path', sparse_dir
    ]

    try:
        subprocess.run(cmd_mapper, check=True)
        print("[SUCCESS] Completed sparse mapping")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Sparse mapping failed: {e}")
        return False


def run_colmap_undistortion(images_dir, sparse_dir, output_dir):
    """Run COLMAP image undistortion."""
    print(f"[INFO] Running image undistortion")

    undistorted_dir = os.path.join(output_dir, 'images_undistorted')
    os.makedirs(undistorted_dir, exist_ok=True)

    cmd_undistort = [
        'colmap',
        'image_undistorter',
        '--image_path', images_dir,
        '--input_path', sparse_dir,
        '--output_path', undistorted_dir,
        '--output_type', 'COLMAP'
    ]

    try:
        
        subprocess.run(cmd_undistort, check=True)
        print("[SUCCESS] Completed image undistortion")

        # Create symlink for backward compatibility (if not exists)
        symlink_path = os.path.join(output_dir, 'images')
        if not os.path.exists(symlink_path):
            os.symlink('images_undistorted', symlink_path)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Image undistortion failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Video to 3DGS Dataset Pipeline - Part 1")
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--max_frames', type=int, default=100,
                        help='Maximum number of frames to extract (30-150)')
    parser.add_argument('--min_frames', type=int, default=30,
                        help='Minimum number of frames to extract (10-100)')
    parser.add_argument('--skip_undistortion', action='store_true',
                        help='Skip image undistortion step')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"[ERROR] Video file not found: {args.video_path}")
        sys.exit(1)

    if args.max_frames < 30 or args.max_frames > 150:
        print("[WARNING] max_frames should be between 30-150, clamping to range")
        args.max_frames = max(30, min(150, args.max_frames))

    if args.min_frames < 10 or args.min_frames > 100:
        print("[WARNING] min_frames should be between 10-100, clamping to range")
        args.min_frames = max(10, min(100, args.min_frames))

    # Ensure min <= max
    if args.min_frames > args.max_frames:
        print(f"[WARNING] min_frames ({args.min_frames}) > max_frames ({args.max_frames}), adjusting")
        args.min_frames = args.max_frames

    # Create output directory structure
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract frames with FFmpeg
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    if not run_ffmpeg_extract_frames(args.video_path, images_dir, args.max_frames, args.min_frames):
        sys.exit(1)

    # Step 2: Run COLMAP SfM
    if not run_colmap_sfm(images_dir, args.output_dir):
        sys.exit(1)

    # Step 3: Run COLMAP undistortion
    if not args.skip_undistortion:
        sparse_dir = os.path.join(args.output_dir, 'sparse', '0')
        if os.path.exists(sparse_dir):
            if not run_colmap_undistortion(images_dir, sparse_dir, args.output_dir):
                sys.exit(1)
        else:
            print("[WARNING] No sparse directory found, skipping undistortion")

    print(f"[SUCCESS] Pipeline completed successfully!")
    print(f"Dataset location: {args.output_dir}")
    print(f"- Images: {images_dir}")
    print(f"- Sparse reconstruction: {os.path.join(args.output_dir, 'sparse')}")
    if not args.skip_undistortion:
        print(f"- Undistorted images: {os.path.join(args.output_dir, 'images_undistorted')}")


if __name__ == "__main__":
    main()