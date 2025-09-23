import os
import subprocess
from typing import Optional

import modal

app = modal.App("omniavatar-train")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        # Install NVIDIA CUDA wheels for torch/vision/audio
        "pip install --upgrade pip",
        "pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(local_path=REPO_ROOT, remote_path="/workspace")
)


# No Mount API required when baking repo into image


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 24)
def train(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    os.chdir("/workspace")
    # Ensure required pretrained weights exist inside the container
    needed = [
        "pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        "pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "pretrained_models/wav2vec2-base-960h",
    ]
    if not all(os.path.exists(p) for p in needed):
        os.makedirs("pretrained_models", exist_ok=True)
        # Base Wan 14B
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "Wan-AI/Wan2.1-T2V-14B",
                "--local-dir",
                "./pretrained_models/Wan2.1-T2V-14B",
            ],
            check=True,
        )
        # Wav2Vec2
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "facebook/wav2vec2-base-960h",
                "--local-dir",
                "./pretrained_models/wav2vec2-base-960h",
            ],
            check=True,
        )
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "scripts/train_lora.py",
        "--config",
        config_path,
    ]
    if extra_hp and len(extra_hp) > 0:
        cmd.extend(["--hp", extra_hp])
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=image, gpu="A100", timeout=60 * 60 * 24)
def train_a100(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    return train.local(config_path=config_path, nproc=nproc, extra_hp=extra_hp)


@app.function(image=image, gpu="H100", timeout=60 * 60 * 24)
def train_h100(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    return train.local(config_path=config_path, nproc=nproc, extra_hp=extra_hp)


if __name__ == "__main__":
    # Run locally with: modal run scripts/modal_train.py::train --config configs/train_lora.yaml --nproc 1
    app.run()
