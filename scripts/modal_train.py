import glob
import os
import subprocess
import tempfile
import urllib.request
from typing import Any, List, Optional

import modal
from modal import fastapi_endpoint

app = modal.App("omniavatar-train")


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VOL_NAME = os.environ.get("OMNIAVATAR_MODAL_VOLUME", "omniavatar-cache")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("fastapi[standard]>=0.110")
    .run_commands(
        # Install matching CUDA wheels without pinning exact versions
        "pip install --upgrade pip",
        # "pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio",
        "pip install torch torchvision torchaudio",
        "pip install deepspeed",
    )
    .add_local_dir(local_path=REPO_ROOT, remote_path="/workspace")
)


# No Mount API required when baking repo into image


def _prepare_inference_env():
    os.chdir("/workspace")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_ALLOW_TF32", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    os.environ.setdefault("HF_HOME", "/vol/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/vol/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/vol/hf")


def _ensure_pretrained_models():
    os.makedirs("/vol/pretrained_models", exist_ok=True)
    needed = [
        "/vol/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/vol/pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        "/vol/pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "/vol/pretrained_models/wav2vec2-base-960h",
        "/vol/pretrained_models/OmniAvatar-14B/pytorch_model.pt",
    ]
    if all(os.path.exists(p) for p in needed):
        return
    subprocess.run(
        [
            "hf",
            "download",
            "Wan-AI/Wan2.1-T2V-14B",
            "--local-dir",
            "/vol/pretrained_models/Wan2.1-T2V-14B",
        ],
        check=True,
    )
    subprocess.run(
        [
            "hf",
            "download",
            "facebook/wav2vec2-base-960h",
            "--local-dir",
            "/vol/pretrained_models/wav2vec2-base-960h",
        ],
        check=True,
    )
    subprocess.run(
        [
            "hf",
            "download",
            "OmniAvatar/OmniAvatar-14B",
            "--local-dir",
            "/vol/pretrained_models/OmniAvatar-14B",
        ],
        check=True,
    )


def _symlink_pretrained_models():
    os.makedirs("/workspace/pretrained_models", exist_ok=True)
    for name in ["Wan2.1-T2V-14B", "wav2vec2-base-960h", "OmniAvatar-14B"]:
        dst = f"/workspace/pretrained_models/{name}"
        src = f"/vol/pretrained_models/{name}"
        if os.path.islink(dst):
            continue
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass


def _copy_outputs_to_volume() -> List[str]:
    saved: List[str] = []
    try:
        out_dir = "/vol/outputs/infer"
        os.makedirs(out_dir, exist_ok=True)
        for mp4 in glob.glob("demo_out/**/*.mp4", recursive=True):
            base = os.path.basename(mp4)
            vol_path = os.path.join(out_dir, base)
            try:
                subprocess.run(["cp", "-f", mp4, vol_path], check=True)
                saved.append(base)
                print(f"[infer] Saved to volume: {vol_path}", flush=True)
                print(
                    f"[infer] Fetch locally with: modal volume get {VOL_NAME} outputs/infer/{base} ./{base}",
                    flush=True,
                )
            except Exception:
                pass
    except Exception:
        pass
    return saved


def _run_inference(
    config_path: str, input_file: str, extra_hp: Optional[str] = None
) -> List[str]:
    _prepare_inference_env()
    _ensure_pretrained_models()
    _symlink_pretrained_models()
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/inference.py",
        "--config",
        config_path,
        "--input_file",
        input_file,
    ]
    if extra_hp:
        cmd.extend(["--hp", extra_hp])
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return _copy_outputs_to_volume()


@app.function(image=image, gpu="A10G", timeout=60 * 60 * 24, volumes={"/vol": vol})
def train(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    os.chdir("/workspace")
    # Improve allocator behavior on large models
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_ALLOW_TF32", "1")
    # Target latest arch when compiling small custom kernels (use 10.0 format for sm_100)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    # Make Hugging Face caches persistent across runs
    os.environ.setdefault("HF_HOME", "/vol/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/vol/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/vol/hf")
    # Expose volume name to training script for fetch hints
    os.environ.setdefault("OMNIAVATAR_MODAL_VOLUME", VOL_NAME)
    # Deepspeed: avoid building CUDA ops inside container
    os.environ.setdefault("DS_BUILD_OPS", "0")
    os.environ.setdefault("DS_BUILD_AIO", "0")
    os.environ.setdefault("DS_BUILD_SPARSE_ATTN", "0")
    os.environ.setdefault("DS_BUILD_UTILS", "0")
    os.environ.setdefault("DS_SKIP_CUDA_CHECK", "1")
    # Diagnostics
    try:
        import torch

        print("[diag] torch:", torch.__version__, "cuda:", torch.version.cuda)
        if torch.cuda.is_available():
            print(
                "[diag] device:",
                torch.cuda.get_device_name(0),
                "cap:",
                torch.cuda.get_device_capability(0),
            )
        try:
            out = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version,name",
                        "--format=csv,noheader",
                    ]
                )
                .decode()
                .strip()
            )
            print("[diag] nvidia-smi:", out)
        except Exception:
            pass
    except Exception:
        pass
    # Ensure required pretrained weights and output dirs exist in persistent volume
    os.makedirs("/vol/pretrained_models", exist_ok=True)
    os.makedirs("/vol/outputs/eval", exist_ok=True)
    vol_needed = [
        "/vol/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/vol/pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        "/vol/pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "/vol/pretrained_models/wav2vec2-base-960h",
        "/vol/pretrained_models/OmniAvatar-14B/pytorch_model.pt",
    ]
    if not all(os.path.exists(p) for p in vol_needed):
        # Download to volume using hf cli
        subprocess.run(
            [
                "hf",
                "download",
                "Wan-AI/Wan2.1-T2V-14B",
                "--local-dir",
                "/vol/pretrained_models/Wan2.1-T2V-14B",
            ],
            check=True,
        )
        subprocess.run(
            [
                "hf",
                "download",
                "facebook/wav2vec2-base-960h",
                "--local-dir",
                "/vol/pretrained_models/wav2vec2-base-960h",
            ],
            check=True,
        )
        # OmniAvatar LoRA
        subprocess.run(
            [
                "hf",
                "download",
                "OmniAvatar/OmniAvatar-14B",
                "--local-dir",
                "/vol/pretrained_models/OmniAvatar-14B",
            ],
            check=True,
        )
    # Symlink workspace paths to volume so existing configs work
    os.makedirs("/workspace/pretrained_models", exist_ok=True)
    if not os.path.islink("/workspace/pretrained_models/Wan2.1-T2V-14B"):
        try:
            os.symlink(
                "/vol/pretrained_models/Wan2.1-T2V-14B",
                "/workspace/pretrained_models/Wan2.1-T2V-14B",
            )
        except FileExistsError:
            pass
    if not os.path.islink("/workspace/pretrained_models/wav2vec2-base-960h"):
        try:
            os.symlink(
                "/vol/pretrained_models/wav2vec2-base-960h",
                "/workspace/pretrained_models/wav2vec2-base-960h",
            )
        except FileExistsError:
            pass
    if not os.path.islink("/workspace/pretrained_models/OmniAvatar-14B"):
        try:
            os.symlink(
                "/vol/pretrained_models/OmniAvatar-14B",
                "/workspace/pretrained_models/OmniAvatar-14B",
            )
        except FileExistsError:
            pass
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


@app.function(image=image, gpu="A100", timeout=60 * 60 * 24, volumes={"/vol": vol})
def train_a100(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    return train.local(config_path=config_path, nproc=nproc, extra_hp=extra_hp)


@app.function(image=image, gpu="H100", timeout=60 * 60 * 24, volumes={"/vol": vol})
def train_h100(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    return train.local(config_path=config_path, nproc=nproc, extra_hp=extra_hp)


@app.function(image=image, gpu="H200", timeout=60 * 60 * 24, volumes={"/vol": vol})
def train_h200(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    return train.local(config_path=config_path, nproc=nproc, extra_hp=extra_hp)


@app.function(image=image, gpu="B200", timeout=60 * 60 * 24, volumes={"/vol": vol})
def train_b200(
    config_path: str = "configs/train_lora.yaml",
    nproc: int = 1,
    extra_hp: Optional[str] = None,
):
    return train.local(config_path=config_path, nproc=nproc, extra_hp=extra_hp)


@app.function(image=image, gpu="B200", timeout=60 * 60 * 4, volumes={"/vol": vol})
def infer_b200(
    config_path: str = "configs/inference.yaml",
    input_file: str = "examples/infer_samples.txt",
):
    os.chdir("/workspace")
    # Same env and caches as training
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_ALLOW_TF32", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
    os.environ.setdefault("HF_HOME", "/vol/hf")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/vol/hf")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/vol/hf")

    # Ensure pretrained weights exist (download if missing)
    os.makedirs("/vol/pretrained_models", exist_ok=True)
    need = [
        "/vol/pretrained_models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/vol/pretrained_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        "/vol/pretrained_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
        "/vol/pretrained_models/wav2vec2-base-960h",
        "/vol/pretrained_models/OmniAvatar-14B/pytorch_model.pt",
    ]
    if not all(os.path.exists(p) for p in need):
        subprocess.run(
            [
                "hf",
                "download",
                "Wan-AI/Wan2.1-T2V-14B",
                "--local-dir",
                "/vol/pretrained_models/Wan2.1-T2V-14B",
            ],
            check=True,
        )
        subprocess.run(
            [
                "hf",
                "download",
                "facebook/wav2vec2-base-960h",
                "--local-dir",
                "/vol/pretrained_models/wav2vec2-base-960h",
            ],
            check=True,
        )
        subprocess.run(
            [
                "hf",
                "download",
                "OmniAvatar/OmniAvatar-14B",
                "--local-dir",
                "/vol/pretrained_models/OmniAvatar-14B",
            ],
            check=True,
        )

    # Symlinks into /workspace
    os.makedirs("/workspace/pretrained_models", exist_ok=True)
    for name in ["Wan2.1-T2V-14B", "wav2vec2-base-960h", "OmniAvatar-14B"]:
        dst = f"/workspace/pretrained_models/{name}"
        src = f"/vol/pretrained_models/{name}"
        if not os.path.islink(dst):
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass

    # Run inference
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "scripts/inference.py",
        "--config",
        config_path,
        "--input_file",
        input_file,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Copy outputs to volume with fetch hints
    saved: List[str] = []
    try:
        out_dir = "/vol/outputs/infer"
        os.makedirs(out_dir, exist_ok=True)
        # Find mp4s under demo_out
        for mp4 in glob.glob("demo_out/**/*.mp4", recursive=True):
            base = os.path.basename(mp4)
            vol_path = os.path.join(out_dir, base)
            try:
                subprocess.run(["cp", "-f", mp4, vol_path], check=True)
                saved.append(base)
                print(f"[infer] Saved to volume: {vol_path}", flush=True)
                print(
                    f"[infer] Fetch locally with: modal volume get {VOL_NAME} outputs/infer/{base} ./{base}",
                    flush=True,
                )
            except Exception:
                pass
    except Exception:
        pass
    return saved


@app.function(image=image, gpu="B200", timeout=60 * 60 * 4, volumes={"/vol": vol})
@fastapi_endpoint(method="POST")
def infer_endpoint(body: dict[str, Any]):
    """
    JSON body:
      {
        "prompt": "...",                # required
        "image_url": "https://...",     # optional
        "audio_url": "https://...",     # optional
        "num_steps": 25,                  # optional
        "guidance_scale": 4.5,            # optional
        "config_path": "configs/inference.yaml"  # optional
      }
    Returns JSON with volume paths and fetch commands.
    """
    try:
        prompt = body.get("prompt", "").strip()
        assert len(prompt) > 0, "prompt is required"
        image_url = body.get("image_url")
        audio_url = body.get("audio_url")
        num_steps = body.get("num_steps")
        guidance_scale = body.get("guidance_scale")
        config_path = body.get("config_path", "configs/inference.yaml")
        disable_lora = body.get("disable_lora", False)

        # Prepare temp workspace and inputs
        workdir = tempfile.mkdtemp(prefix="infer_")
        image_path = None
        audio_path = None
        if image_url:
            image_path = os.path.join(workdir, "ref.jpg")
            urllib.request.urlretrieve(image_url, image_path)
        if audio_url:
            audio_path = os.path.join(workdir, "audio.wav")
            urllib.request.urlretrieve(audio_url, audio_path)

        # Build a one-line manifest
        parts = [prompt]
        if image_path is not None:
            parts.append(image_path)
            if audio_path is not None:
                parts.append(audio_path)
        manifest_line = "@@".join(parts)
        manifest_path = os.path.join(workdir, "input.txt")
        with open(manifest_path, "w") as f:
            f.write(manifest_line + "\n")

        # Kick off GPU inference synchronously in a GPU container
        hp_override = None
        if disable_lora:
            hp_override = "train_architecture=none,disable_lora=true"
        saved = _run_inference(
            config_path=config_path,
            input_file=manifest_path,
            extra_hp=hp_override,
        )

        # Build fetch commands
        fetch = [
            f"modal volume get {VOL_NAME} outputs/infer/{name} ./{name}"
            for name in (saved or [])
        ]
        return {
            "ok": True,
            "saved_files": [f"outputs/infer/{name}" for name in (saved or [])],
            "volume": VOL_NAME,
            "fetch": fetch,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.function(image=image, gpu="B200", timeout=60 * 60 * 4, volumes={"/vol": vol})
@fastapi_endpoint(method="GET")
def smoke_test(index: int = 1, config_path: str = "configs/inference.yaml"):
    """
    Runs a quick inference on one line from examples/infer_samples.txt.
    - index: 1-10 (default 1)
    - config_path: optional config override
    Returns JSON with saved filenames on the Modal volume and fetch commands.
    """
    try:
        os.chdir("/workspace")
        src_file = "examples/infer_samples.txt"
        assert os.path.exists(src_file), f"{src_file} not found"

        with open(src_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if len(lines) == 0:
            return {"ok": False, "error": "no examples in examples/infer_samples.txt"}

        # Clamp index into range 1..len(lines)
        if index is None:
            index = 1
        try:
            index = int(index)
        except Exception:
            index = 1
        index = max(1, min(len(lines), index))
        chosen = lines[index - 1]

        # Write a temporary manifest containing only the chosen line
        workdir = tempfile.mkdtemp(prefix="smoke_")
        manifest_path = os.path.join(workdir, "input.txt")
        with open(manifest_path, "w") as f:
            f.write(chosen + "\n")

        saved = _run_inference(
            config_path=config_path,
            input_file=manifest_path,
        )
        fetch = [
            f"modal volume get {VOL_NAME} outputs/infer/{name} ./{name}"
            for name in (saved or [])
        ]
        return {
            "ok": True,
            "used_example_index": index,
            "saved_files": [f"outputs/infer/{name}" for name in (saved or [])],
            "volume": VOL_NAME,
            "fetch": fetch,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    # Run locally with: modal run scripts/modal_train.py::train --config configs/train_lora.yaml --nproc 1
    app.run()
