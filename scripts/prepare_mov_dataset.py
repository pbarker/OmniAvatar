import argparse
import os
import subprocess
from glob import glob
from pathlib import Path


def run_command(command: list[str]):
    subprocess.run(command, check=True)


def prepare_dataset(
    input_dir: str,
    output_root: str,
    fps: int = 25,
    sample_rate: int = 16000,
    segment_seconds: float = 5.0,
    segment_frames: int | None = None,
    prompt: str = "A realistic video of a person speaking to the camera",
    manifest_path: str | None = None,
):
    input_dir = os.path.abspath(input_dir)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    clips_dir = os.path.join(output_root, "clips")
    frames_dir = os.path.join(output_root, "frames")
    audio_dir = os.path.join(output_root, "audio")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    if manifest_path is None:
        manifest_path = os.path.join(output_root, "train_manifest.txt")
    else:
        manifest_path = os.path.abspath(manifest_path)
    # truncate/create manifest
    open(manifest_path, "w").close()

    # Determine desired clip duration (default 5 seconds)
    segment_time = (
        segment_seconds
        if segment_seconds is not None
        else ((segment_frames or 57) / float(fps))
    )

    mov_files: list[str] = []
    for ext in ("*.mov", "*.MOV"):
        mov_files.extend(glob(os.path.join(input_dir, "**", ext), recursive=True))
    mov_files = sorted(mov_files)

    for mov_path in mov_files:
        base = Path(mov_path).stem
        normalized = os.path.join(output_root, f"{base}_norm.mp4")
        # 1) normalize fps to target (keep audio)
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                mov_path,
                "-vf",
                f"fps={fps}",
                normalized,
            ]
        )
        # 2) trim to desired seconds (single clip per source)
        clip_out = os.path.join(clips_dir, f"{base}.mp4")
        run_command(
            [
                "ffmpeg",
                "-y",
                "-i",
                normalized,
                "-t",
                f"{segment_time:.3f}",
                "-c",
                "copy",
                clip_out,
            ]
        )

        # 3) per-clip: first frame and 16k audio; append manifest
        clip_paths = [clip_out]
        for clip_path in clip_paths:
            clip_id = Path(clip_path).stem
            frame_out = os.path.join(frames_dir, f"{clip_id}.jpg")
            audio_out = os.path.join(audio_dir, f"{clip_id}.wav")

            # first frame (reference)
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    clip_path,
                    "-vf",
                    "select=eq(n\\,0)",
                    "-q:v",
                    "2",
                    frame_out,
                ]
            )

            # audio 16k mono
            run_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    clip_path,
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-vn",
                    audio_out,
                ]
            )

            # append manifest (use repo-relative paths if output_root is inside repo)
            rel_frame = os.path.relpath(frame_out, start=os.getcwd())
            rel_audio = os.path.relpath(audio_out, start=os.getcwd())
            rel_clip = os.path.relpath(clip_path, start=os.getcwd())
            with open(manifest_path, "a") as mf:
                mf.write(f"{prompt}@@{rel_frame}@@{rel_audio}@@{rel_clip}\n")

    print(f"Prepared dataset under: {output_root}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare .mov videos into training manifest and assets"
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing .mov files (recursive)"
    )
    parser.add_argument(
        "output_root",
        type=str,
        nargs="?",
        default="data",
        help="Output root directory (default: data)",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--segment_frames",
        type=int,
        default=57,
        help="Frames per segment (default matches 30k token budget at 480p)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A realistic video of a person speaking to the camera",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to output manifest (default: output_root/train_manifest.txt)",
    )
    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input_dir,
        output_root=args.output_root,
        fps=args.fps,
        sample_rate=args.sample_rate,
        segment_frames=args.segment_frames,
        prompt=args.prompt,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
