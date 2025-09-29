import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, write_video
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from peft import LoraConfig, inject_adapter_in_model
from transformers import Wav2Vec2FeatureExtractor

from OmniAvatar.models.model_manager import ModelManager
from OmniAvatar.models.wav2vec import Wav2VecModel
from OmniAvatar.utils.args_config import parse_args
from OmniAvatar.wan_video import WanVideoPipeline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    """
    Each line in manifest: "prompt@@ref_image_path@@audio_path@@video_path"
    This dataset only returns raw paths and prompt; encoding is done inside the training loop
    to share the same VAE/audio/text encoders as inference/pipeline.
    """

    def __init__(self, manifest_path: str):
        super().__init__()
        with open(manifest_path, "r") as f:
            self.lines = [line.strip() for line in f if len(line.strip()) > 0]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        parts = self.lines[idx].split("@@")
        assert len(parts) == 4, "Each line must be: prompt@@ref_image@@audio@@video"
        prompt, ref_img, audio, video = parts
        return dict(prompt=prompt, ref_img=ref_img, audio=audio, video=video)


def match_size(image_size, h, w):
    ratio_ = 9999
    size_ = 9999
    select_size = None
    for image_s in image_size:
        ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
        size_tmp = abs(max(image_s) - max(w, h))
        if ratio_tmp < ratio_:
            ratio_ = ratio_tmp
            size_ = size_tmp
            select_size = image_s
        if ratio_ == ratio_tmp:
            if size_ == size_tmp:
                select_size = image_s
    return select_size


def resize_pad_tensor(video_tensor: torch.Tensor, tgt_size: Tuple[int, int]):
    # video_tensor: [T, H, W, C] uint8 0..255
    _, H, W, _ = video_tensor.shape
    h, w = H, W
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)
    video = video_tensor.permute(0, 3, 1, 2).float() / 255.0  # T, C, H, W in [0,1]
    video = F.interpolate(
        video, size=(scale_h, scale_w), mode="bilinear", align_corners=False
    )

    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left
    video = F.pad(
        video, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )
    # -> T, C, H, W in [0,1]
    return video


def prepare_audio_embeddings(
    audio_path: str,
    wav_feature_extractor,
    audio_encoder: Wav2VecModel,
    sample_rate: int,
    fps: int,
    device,
    torch_dtype,
):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    input_values = np.squeeze(
        wav_feature_extractor(audio, sampling_rate=sample_rate).input_values
    )
    # Run Wav2Vec on CPU to conserve GPU memory, then move result to GPU
    input_values = torch.from_numpy(input_values).float().to(device="cpu")
    audio_len_frames = math.ceil(len(input_values) / sample_rate * fps)
    input_values = input_values.unsqueeze(0)
    with torch.no_grad():
        hidden_states = audio_encoder(
            input_values,
            seq_len=audio_len_frames,
            output_hidden_states=True,
            return_dict=True,
        )
        audio_embeddings = hidden_states.last_hidden_state
        for mid_hidden_states in hidden_states.hidden_states:
            audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
    audio_embeddings = audio_embeddings.squeeze(0)  # [T, feat]
    return audio_embeddings.to(device=device, dtype=torch_dtype), audio_len_frames


class CheckpointBlock(torch.nn.Module):
    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, context, t_mod, freqs):
        def _fn(x_in, context_in, t_mod_in, freqs_in):
            return self.block(x_in, context_in, t_mod_in, freqs_in)

        return checkpoint(_fn, x, context, t_mod, freqs, use_reentrant=False)


def build_image_prefix(
    pipe: WanVideoPipeline,
    ref_image_path: str,
    T_lat: int,
    height: int,
    width: int,
    dtype,
    device,
):
    image = Image.open(ref_image_path).convert("RGB")
    image = np.asarray(image).astype(np.uint8)
    # format to tensor: 1, 3, 1, H, W in [-1,1]
    img = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img = F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)
    img = img[:, :, None]
    img = img.to(device=device, dtype=dtype)
    img = img * 2.0 - 1.0

    pipe.load_models_to_device(["vae"])
    img_lat = pipe.encode_video(img)
    msk = torch.zeros_like(img_lat.repeat(1, 1, T_lat, 1, 1)[:, :1])
    image_cat = img_lat.repeat(1, 1, T_lat, 1, 1)
    msk[:, :, 1:] = 1
    y = torch.cat([image_cat, msk], dim=1)
    return {"y": y}


def _hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def load_or_build_prompt_embed(
    cache_dir: Optional[str], pipe: WanVideoPipeline, prompt: str, torch_dtype
):
    if cache_dir is None:
        return pipe.encode_prompt(prompt, positive=True)
    os.makedirs(cache_dir, exist_ok=True)
    key = _hash_key("prompt", prompt)
    path = os.path.join(cache_dir, f"{key}.prompt.pt")
    if os.path.exists(path):
        data = torch.load(path, map_location="cpu")
        out = {"context": data.to(dtype=torch_dtype, device=pipe.device)}
        return out
    out = pipe.encode_prompt(prompt, positive=True)
    torch.save(out["context"].to(torch.float32).cpu(), path)
    return out


def load_or_build_video_latents(
    cache_dir: Optional[str],
    pipe: WanVideoPipeline,
    video_path: str,
    height: int,
    width: int,
    torch_dtype,
):
    if cache_dir is None:
        # raw load and encode without caching
        vid_frames, _, _ = read_video(video_path, pts_unit="sec")
        video_tensor = resize_pad_tensor(
            vid_frames, (height, width)
        )  # T,C,H,W in [0,1]
        video_tensor = video_tensor.unsqueeze(0).to(
            device=pipe.device, dtype=torch_dtype
        )
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
        video_tensor = video_tensor * 2.0 - 1.0
        pipe.load_models_to_device(["vae"])
        if hasattr(pipe, "vae"):
            pipe.vae.to(device=pipe.device)
        with torch.no_grad():
            lat = pipe.encode_video(video_tensor)
        return lat
    os.makedirs(cache_dir, exist_ok=True)
    key = _hash_key("video", video_path, f"{height}x{width}")
    path = os.path.join(cache_dir, f"{key}.latents.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=pipe.device).to(dtype=torch_dtype)
    vid_frames, _, _ = read_video(video_path, pts_unit="sec")
    video_tensor = resize_pad_tensor(vid_frames, (height, width))
    video_tensor = video_tensor.unsqueeze(0).to(device=pipe.device, dtype=torch_dtype)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
    video_tensor = video_tensor * 2.0 - 1.0
    pipe.load_models_to_device(["vae"])
    if hasattr(pipe, "vae"):
        pipe.vae.to(device=pipe.device)
    with torch.no_grad():
        lat = pipe.encode_video(video_tensor)
    torch.save(lat.to(torch.float16).cpu(), path)
    return lat


def load_or_build_image_latent(
    cache_dir: Optional[str],
    pipe: WanVideoPipeline,
    image_path: str,
    height: int,
    width: int,
    torch_dtype,
):
    if cache_dir is None:
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image).astype(np.uint8)
        img = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img = F.interpolate(
            img, size=(height, width), mode="bilinear", align_corners=False
        )
        img = img[:, :, None]
        img = img.to(device=pipe.device, dtype=torch_dtype)
        img = img * 2.0 - 1.0
        pipe.load_models_to_device(["vae"])
        img_lat = pipe.encode_video(img)
        return img_lat
    os.makedirs(cache_dir, exist_ok=True)
    key = _hash_key("image", image_path, f"{height}x{width}")
    path = os.path.join(cache_dir, f"{key}.imglat.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=pipe.device).to(dtype=torch_dtype)
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image).astype(np.uint8)
    img = torch.as_tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img = F.interpolate(img, size=(height, width), mode="bilinear", align_corners=False)
    img = img[:, :, None]
    img = img.to(device=pipe.device, dtype=torch_dtype)
    img = img * 2.0 - 1.0
    pipe.load_models_to_device(["vae"])
    img_lat = pipe.encode_video(img)
    torch.save(img_lat.to(torch.float16).cpu(), path)
    return img_lat


def save_tensor_video_as_mp4(path: str, frames_tensor: torch.Tensor, fps: int):
    # frames_tensor: [T, H, W, C] in [0,1]
    frames_uint8 = (frames_tensor.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
    write_video(path, frames_uint8, fps=fps, video_codec="h264", options={"crf": "18"})


class _DummyPbar:
    def __init__(self):
        pass

    def write(self, msg: str):
        print(msg, flush=True)

    def update(self, n: int):
        pass

    def set_description(self, msg: str):
        print(msg, flush=True)

    def close(self):
        pass


def main():
    args = parse_args()
    set_seed(
        args.get("seed", 42) if isinstance(args, dict) else getattr(args, "seed", 42)
    )

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(dist.get_rank())
    device = torch.device(f"cuda:{dist.get_rank()}")

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load base models
    model_manager = ModelManager(device="cpu", infer=True)
    model_manager.load_models(
        [
            args.dit_path.split(","),
            args.text_encoder_path,
            args.vae_path,
        ],
        torch_dtype=torch_dtype,
        device="cpu",
    )

    pipe = WanVideoPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch_dtype,
        device=str(device),
        use_usp=False,
        infer=False,
    )

    # Optional VRAM management: offload inactive modules to CPU
    if bool(getattr(args, "enable_vram_management", True)):
        # keep a small persistent window in DiT to reduce thrash
        pipe.enable_vram_management(
            num_persistent_param_in_dit=int(
                getattr(args, "num_persistent_param_in_dit", 0)
            )
        )

    # Inject LoRA into DiT and select trainable params
    assert (
        getattr(args, "train_architecture", "lora") == "lora"
    ), "This script currently supports LoRA fine-tuning only."
    # Resolve LoRA hyperparams; prefer checkpoint config if present
    lora_ckpt = getattr(args, "lora_checkpoint", None)
    allow_new_lora = bool(getattr(args, "allow_new_lora", False))
    if not lora_ckpt:
        for candidate in [
            os.path.join(
                "checkpoints", "checkpoints", "OmniAvatar-LoRA-14B", "pytorch_model.pt"
            ),
            os.path.join("checkpoints", "OmniAvatar-LoRA-14B", "pytorch_model.pt"),
        ]:
            if os.path.exists(candidate):
                lora_ckpt = candidate
                break
    lora_rank = int(getattr(args, "lora_rank", 128))
    lora_alpha = int(getattr(args, "lora_alpha", 64))
    lora_target_modules = getattr(args, "lora_target_modules", "q,k,v,o,ffn.0,ffn.2")
    if lora_ckpt and os.path.exists(lora_ckpt):
        ckpt_dir = os.path.dirname(lora_ckpt)
        cfg_path = os.path.join(ckpt_dir, "config.json")
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    ck_cfg = json.load(f)
                lora_rank = int(ck_cfg.get("lora_rank", lora_rank))
                lora_alpha = int(ck_cfg.get("lora_alpha", lora_alpha))
                lora_target_modules = ck_cfg.get(
                    "lora_target_modules", lora_target_modules
                )
            else:
                st = torch.load(lora_ckpt, map_location="cpu")
                for k, v in st.items():
                    if k.endswith("lora_A.default.weight") and v.ndim == 2:
                        lora_rank = int(v.shape[0])
                        break
        except Exception:
            pass

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=lora_target_modules.split(","),
    )
    dit = inject_adapter_in_model(lora_cfg, pipe.denoising_model())
    if lora_ckpt and os.path.exists(lora_ckpt):
        state = torch.load(lora_ckpt, map_location="cpu")
        _ = dit.load_state_dict(
            {k: v for k, v in state.items() if "lora" in k.lower()}, strict=False
        )
    else:
        if not allow_new_lora:
            raise RuntimeError(
                "Required base LoRA not found. Set --hp lora_checkpoint=/path/to/pytorch_model.pt "
                "or pass --hp allow_new_lora=true to train a new LoRA from scratch."
            )

    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)
    # Freeze base DiT weights; keep LoRA params trainable
    for name, p in dit.named_parameters():
        if "lora" in name.lower():
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

    pipe.dit = dit.train().to(device)
    # Optional gradient checkpointing across DiT blocks
    if bool(getattr(args, "enable_grad_ckpt", True)):
        wrapped_blocks = []
        for blk in pipe.dit.blocks:
            wrapped_blocks.append(CheckpointBlock(blk))
        pipe.dit.blocks = torch.nn.ModuleList(wrapped_blocks)

    # Audio encoder (frozen)
    wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    audio_encoder = Wav2VecModel.from_pretrained(
        args.wav2vec_path, local_files_only=True
    ).to(device="cpu")
    audio_encoder.feature_extractor._freeze_parameters()
    for p in audio_encoder.parameters():
        p.requires_grad_(False)

    # Data
    train_dataset = ManifestDataset(getattr(args, "train_manifest"))
    # Holdout the first entry for evaluation
    holdout_line = train_dataset.lines[0] if len(train_dataset.lines) > 0 else None
    holdout_prompt = holdout_ref = holdout_audio = holdout_video = None
    if holdout_line is not None:
        parts = holdout_line.split("@@")
        if len(parts) == 4:
            holdout_prompt, holdout_ref, holdout_audio, holdout_video = parts

    # Print dataset overview and paths (rank 0 only)
    if dist.get_rank() == 0:
        try:
            manifest_path = getattr(args, "train_manifest")
            print(
                f"[data] Loaded manifest with {len(train_dataset)} items: {manifest_path}",
                flush=True,
            )
            if holdout_line is not None:
                print(
                    "[data] HOLDOUT (eval):\n"
                    f'        prompt="{holdout_prompt}"\n'
                    f'        ref="{os.path.abspath(holdout_ref)}"\n'
                    f'        audio="{os.path.abspath(holdout_audio)}"\n'
                    f'        video="{os.path.abspath(holdout_video)}"',
                    flush=True,
                )
            print("[data] Training items:", flush=True)
            for idx, line in enumerate(train_dataset.lines):
                parts_i = line.split("@@")
                if len(parts_i) != 4:
                    continue
                p_i, r_i, a_i, v_i = parts_i
                tag = (
                    "[HOLDOUT]"
                    if holdout_line is not None and line == holdout_line
                    else "         "
                )
                print(
                    f'[data] {idx:05d} {tag} prompt="{p_i}" ref="{os.path.abspath(r_i)}" audio="{os.path.abspath(a_i)}" video="{os.path.abspath(v_i)}"',
                    flush=True,
                )
        except Exception:
            pass
    batch_size = int(getattr(args, "batch_size", 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    lr = float(getattr(args, "lr", 5e-5))
    params_to_train = [p for p in dit.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=0.0)

    # Optional DeepSpeed ZeRO Offload
    use_deepspeed = bool(getattr(args, "use_deepspeed", False))
    if use_deepspeed:
        try:
            import deepspeed  # type: ignore
        except Exception as e:
            print(
                f"[warn] Deepspeed unavailable ({e}); falling back to standard optimizer."
            )
            use_deepspeed = False

        ds_stage = int(getattr(args, "ds_stage", 2))
        offload_params = bool(getattr(args, "ds_offload_params", True))
        offload_optimizer = bool(getattr(args, "ds_offload_optimizer", True))
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": int(getattr(args, "grad_accum_steps", 1)),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": lr,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            },
            "bf16": {"enabled": args.dtype == "bf16"},
            "zero_optimization": {
                "stage": ds_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "sub_group_size": 1e9,
                "offload_param": {"device": "cpu", "pin_memory": True}
                if offload_params and ds_stage >= 2
                else {"device": "none"},
                "offload_optimizer": {"device": "cpu", "pin_memory": True}
                if offload_optimizer and ds_stage >= 2
                else {"device": "none"},
            },
        }
        try:
            pipe.dit, opt, _, _ = deepspeed.initialize(
                model=pipe.dit,
                model_parameters=params_to_train,
                optimizer=opt,
                config=ds_config,
            )
            model_for_step = pipe.dit
        except Exception as e:
            print(f"[warn] Deepspeed init failed ({e}); using standard optimizer.")
            use_deepspeed = False
            model_for_step = pipe.dit
    else:
        model_for_step = pipe.dit

    # Scheduler for training targets/weights
    pipe.scheduler.set_timesteps(
        num_inference_steps=int(getattr(args, "num_train_timesteps", 1000)),
        training=True,
    )

    # Train config
    audio_cfg_dropout = float(getattr(args, "audio_cfg_dropout", 0.1))
    max_tokens = int(getattr(args, "max_tokens", 30000))
    fps = int(getattr(args, "fps", 25))
    sample_rate = int(getattr(args, "sample_rate", 16000))
    max_hw = int(getattr(args, "max_hw", 720))
    prefix_lat_min = int(getattr(args, "prefix_latent_min", 1))
    prefix_lat_max = int(getattr(args, "prefix_latent_max", 4))
    cache_dir = getattr(args, "cache_dir", None)
    phase_two_prob = float(getattr(args, "phase_two_highres_prob", 0.5))
    phase_two_start = int(getattr(args, "phase_two_start_step", 6000))
    # Diagnostics / preview controls
    smi_every = int(getattr(args, "smi_every", 0))  # 0 disables periodic nvidia-smi
    preview_timesteps_default = int(getattr(args, "preview_timesteps", 8))
    preview_timesteps_eval = int(
        getattr(args, "preview_timesteps_eval", preview_timesteps_default)
    )
    eval_loss_samples = int(
        getattr(args, "eval_loss_samples", 1)
    )  # average N eval losses
    smi_every = int(getattr(args, "smi_every", 0))  # 0 disables periodic nvidia-smi
    train_on_holdout = bool(getattr(args, "train_on_holdout", False))

    total_steps = int(getattr(args, "num_train_steps", 10000))
    save_every = int(getattr(args, "save_every", 1000))

    exp_dir = (
        args.exp_path
        if os.path.isabs(args.exp_path)
        else os.path.join("checkpoints", args.exp_path)
    )
    os.makedirs(exp_dir, exist_ok=True)

    # -------- Pre-flight previews before training --------
    def _save_and_copy(frames_eval: Optional[torch.Tensor], out_dir: str, name: str):
        os.makedirs(out_dir, exist_ok=True)
        out_mp4 = os.path.join(out_dir, f"{name}.mp4")
        size_bytes = -1
        try:
            if frames_eval is not None:
                save_tensor_video_as_mp4(out_mp4, frames_eval, fps)
                if os.path.exists(out_mp4):
                    size_bytes = os.path.getsize(out_mp4)
        except Exception:
            pass
        try:
            pbar.write(
                f"[preflight] Saved video to {out_mp4} (size={size_bytes} bytes)"
            )
        except Exception:
            print(
                f"[preflight] Saved video to {out_mp4} (size={size_bytes} bytes)",
                flush=True,
            )
        # Copy to Modal volume
        try:
            vol_dir = os.path.join("/vol", "outputs", "infer")
            os.makedirs(vol_dir, exist_ok=True)
            vol_mp4 = os.path.join(vol_dir, f"{name}.mp4")
            if os.path.exists(out_mp4) and size_bytes > 0:
                shutil.copyfile(out_mp4, vol_mp4)
            vol_size = os.path.getsize(vol_mp4) if os.path.exists(vol_mp4) else -1
            vol_name = os.environ.get("OMNIAVATAR_MODAL_VOLUME", "omniavatar-cache")
            fetch_cmd = (
                f"modal volume get {vol_name} outputs/infer/{name}.mp4 ./{name}.mp4"
            )
            try:
                pbar.write(
                    f"[preflight] Also saved to volume: {vol_mp4} (size={vol_size} bytes)"
                )
                pbar.write(f"[preflight] Fetch locally with: {fetch_cmd}")
            except Exception:
                print(
                    f"[preflight] Also saved to volume: {vol_mp4} (size={vol_size} bytes)",
                    flush=True,
                )
                print(f"[preflight] Fetch locally with: {fetch_cmd}", flush=True)
        except Exception:
            pass

    def _run_preview(prompt: str, image_path: str, audio_path: str, tag: str):
        try:
            # Pick resolution
            try:
                image = Image.open(image_path).convert("RGB")
                Hh, Ww = image.size[1], image.size[0]
            except Exception:
                Hh, Ww = 720, 720
            image_sizes = getattr(args, "image_sizes_720")
            e_height, e_width = match_size(image_sizes, Hh, Ww)

            # Build latents backbone by encoding a zero video of needed T
            L_frames = max(5, int(max_tokens * 16 * 16 * 4 / e_height / e_width))
            L_frames = (L_frames // 4 * 4 + 1) if L_frames % 4 != 0 else (L_frames - 3)
            T_lat = (L_frames + 3) // 4
            zero_vid = torch.zeros(1, 3, L_frames, e_height, e_width)
            zero_vid = zero_vid.to(device=pipe.device, dtype=torch_dtype) * 2.0 - 1.0
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                base_lat = pipe.encode_video(zero_vid)

            # Image prefix
            img_lat = load_or_build_image_latent(
                cache_dir, pipe, image_path, e_height, e_width, torch_dtype
            )
            msk = torch.zeros_like(img_lat.repeat(1, 1, base_lat.shape[2], 1, 1)[:, :1])
            image_cat = img_lat.repeat(1, 1, base_lat.shape[2], 1, 1)
            msk[:, :, 1:] = 1
            image_emb_eval = {"y": torch.cat([image_cat, msk], dim=1)}

            # Audio
            audio_emb_eval, audio_len_frames_eval = prepare_audio_embeddings(
                audio_path,
                wav_feature_extractor,
                audio_encoder,
                sample_rate,
                fps,
                device,
                torch_dtype,
            )
            first_fixed_frame = 1
            need = L_frames - first_fixed_frame
            if audio_len_frames_eval < need:
                audio_len_frames_eval = audio_len_frames_eval + (
                    need - audio_len_frames_eval % need
                )
            elif (audio_len_frames_eval - need) % (L_frames - 1) != 0:
                audio_len_frames_eval = audio_len_frames_eval + (
                    (L_frames - 1) - (audio_len_frames_eval - need) % (L_frames - 1)
                )
            audio_emb_eval = F.pad(
                audio_emb_eval,
                (0, 0, 0, audio_len_frames_eval - audio_emb_eval.shape[0]),
                mode="constant",
                value=0,
            )
            audio_prefix = torch.zeros_like(audio_emb_eval[:first_fixed_frame])
            audio_tensor_eval = torch.cat([audio_prefix, audio_emb_eval[:need]], dim=0)
            audio_tensor_eval = audio_tensor_eval.unsqueeze(0).to(
                device=device, dtype=torch_dtype
            )
            audio_cond_eval = {"audio_emb": audio_tensor_eval}

            # Prompt
            pipe.load_models_to_device(["text_encoder"])
            prompt_emb_eval = pipe.encode_prompt(prompt, positive=True)

            # Short preview
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            steps = max(1, int(preview_timesteps_eval))
            pipe.scheduler.set_timesteps(num_inference_steps=steps, training=False)
            latents_eval = torch.randn_like(base_lat).to(
                device=pipe.device, dtype=torch_dtype
            )
            pipe.load_models_to_device(["dit"])
            prev_training_mode = pipe.dit.training
            pipe.dit.eval()
            with torch.no_grad():
                for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
                    timestep = timestep.unsqueeze(0).to(
                        dtype=torch_dtype, device=pipe.device
                    )
                    pred_eval = pipe.dit(
                        latents_eval,
                        timestep=timestep,
                        **prompt_emb_eval,
                        **image_emb_eval,
                        **audio_cond_eval,
                    )
                    latents_eval = pipe.scheduler.step(
                        pred_eval, pipe.scheduler.timesteps[progress_id], latents_eval
                    )
            pipe.dit.train(prev_training_mode)
            # Decode
            frames_eval = None
            try:
                pipe.load_models_to_device(["vae"])
                with torch.no_grad():
                    frames_eval = pipe.decode_video(latents_eval)
            except Exception:
                pass
            if frames_eval is not None:
                frames_eval = (frames_eval[0].permute(1, 2, 3, 0).float() + 1) / 2
            _save_and_copy(frames_eval, os.path.join(exp_dir, "preflight"), f"{tag}")
        except Exception:
            err = traceback.format_exc()
            try:
                pbar.write(f"[preflight] ERROR during {tag} preview")
                pbar.write(err)
            except Exception:
                print(f"[preflight] ERROR during {tag} preview", flush=True)
                print(err, flush=True)

    if dist.get_rank() == 0:
        # 1) Preview on a known example from ./examples if available
        examples_file = getattr(
            args, "examples_input_file", "examples/infer_samples.txt"
        )
        try:
            if os.path.exists(examples_file):
                with open(examples_file, "r") as f:
                    line = f.readline().strip()
                parts_e = line.split("@@")
                if len(parts_e) == 3:
                    ex_prompt, ex_img, ex_audio = parts_e
                    _run_preview(ex_prompt, ex_img, ex_audio, tag="preflight_example")
        except Exception:
            pass
        # 2) Preview on the eval holdout sample if present
        try:
            if (
                holdout_line is not None
                and holdout_prompt
                and holdout_ref
                and holdout_audio
            ):
                _run_preview(
                    holdout_prompt, holdout_ref, holdout_audio, tag="preflight_holdout"
                )
        except Exception:
            pass

    step = 0
    ema_loss = None
    best_ema = None
    no_improve_steps = 0
    ema_decay = float(getattr(args, "loss_ema_decay", 0.98))
    early_stop_patience = int(getattr(args, "early_stop_patience", 400))
    early_stop_patience_evals = int(getattr(args, "early_stop_patience_evals", 0))
    early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
    stop_training = False
    # Track best eval loss and when it occurred for early stopping
    best_eval = None
    best_eval_step = -1
    eval_no_improve = 0
    dit.train()
    # Use plain printing on rank0 to avoid log overwrite by tqdm
    if dist.get_rank() == 0:
        pbar = _DummyPbar()
    else:
        pbar = tqdm(total=total_steps, disable=True)

    while step < total_steps and not stop_training:
        for batch in train_loader:
            if step >= total_steps:
                break
            if stop_training:
                break
            prompt_list = batch["prompt"]
            ref_img_list = batch["ref_img"]
            audio_list = batch["audio"]
            video_list = batch["video"]

            # For simplicity, operate per-sample (effective batch size 1). Accumulation can be added later.
            opt.zero_grad(set_to_none=True)

            # Skip held-out sample for training (unless explicitly allowed)
            if (
                not train_on_holdout
                and holdout_video is not None
                and video_list[0] == holdout_video
            ):
                continue

            # Prepare prompt
            prompt = prompt_list[0]
            # Ensure text encoder is on the correct device before encoding
            pipe.load_models_to_device(["text_encoder"])
            # Double-ensure the module is on GPU to avoid CPU/CUDA mismatch
            if hasattr(pipe, "text_encoder"):
                pipe.text_encoder.to(device=pipe.device)
            prompt_emb = pipe.encode_prompt(prompt, positive=True)

            # Resolution schedule and cached latents
            use_highres = False
            if (
                step >= phase_two_start
                and max_hw == 1280
                and random.random() < phase_two_prob
            ):
                use_highres = True
            image_sizes_key = f"image_sizes_{1280 if use_highres else 720}"
            image_sizes = getattr(args, image_sizes_key)
            # Peek H,W to choose target size
            vid_frames, _, _ = read_video(video_list[0], pts_unit="sec")
            H, W = vid_frames.shape[1], vid_frames.shape[2]
            height, width = match_size(image_sizes, H, W)
            clean_latents = load_or_build_video_latents(
                cache_dir, pipe, video_list[0], height, width, torch_dtype
            )

            # Determine token budget -> number of video frames
            # Ensure resulting latent shape aligns with 4-frame groupings and +1 rule
            num_video_frames = max(5, int(max_tokens * 16 * 16 * 4 / height / width))
            num_video_frames = (
                num_video_frames // 4 * 4 + 1
                if num_video_frames % 4 != 0
                else num_video_frames - 3
            )
            num_latent_frames = (num_video_frames + 3) // 4
            # Trim/pad clean_latents to T_lat so spatial dims match
            cur_T = clean_latents.shape[2]
            if cur_T >= num_latent_frames:
                clean_latents = clean_latents[:, :, :num_latent_frames]
            else:
                pad = num_latent_frames - cur_T
                clean_latents = torch.cat(
                    [
                        clean_latents,
                        torch.zeros_like(clean_latents[:, :, :1]).repeat(
                            1, 1, pad, 1, 1
                        ),
                    ],
                    dim=2,
                )

            # Ensure target has same spatial dims as prediction
            clean_latents = clean_latents.contiguous()

            # Build image prefix embedding (random prefix length 1..4 latent frames)
            prefix_lat = random.randint(prefix_lat_min, prefix_lat_max)
            # Build image prefix (can be cached by image path)
            img_lat = load_or_build_image_latent(
                cache_dir, pipe, ref_img_list[0], height, width, torch_dtype
            )
            msk = torch.zeros_like(img_lat.repeat(1, 1, num_latent_frames, 1, 1)[:, :1])
            image_cat = img_lat.repeat(1, 1, num_latent_frames, 1, 1)
            msk[:, :, 1:] = 1
            image_emb = {"y": torch.cat([image_cat, msk], dim=1)}

            # Prepare audio embeddings and optional CFG dropout
            audio_embeddings, audio_len_frames = prepare_audio_embeddings(
                audio_list[0],
                wav_feature_extractor,
                audio_encoder,
                sample_rate,
                fps,
                device,
                torch_dtype,
            )
            # Pad/trim audio to match L, then build tensor with prefix alignment like inference
            first_fixed_frame = 1  # training uses reference frame as first fixed frame
            if audio_len_frames < num_video_frames - first_fixed_frame:
                audio_len_frames = audio_len_frames + (
                    (num_video_frames - first_fixed_frame)
                    - audio_len_frames % (num_video_frames - first_fixed_frame)
                )
            elif (audio_len_frames - (num_video_frames - first_fixed_frame)) % (
                num_video_frames - 1
            ) != 0:
                audio_len_frames = audio_len_frames + (
                    (num_video_frames - 1)
                    - (audio_len_frames - (num_video_frames - first_fixed_frame))
                    % (num_video_frames - 1)
                )
            audio_embeddings = F.pad(
                audio_embeddings,
                (0, 0, 0, audio_len_frames - audio_embeddings.shape[0]),
                mode="constant",
                value=0,
            )
            audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
            audio_tensor = torch.cat(
                [
                    audio_prefix,
                    audio_embeddings[: num_video_frames - first_fixed_frame],
                ],
                dim=0,
            )
            audio_tensor = audio_tensor.unsqueeze(0).to(
                device=device, dtype=torch_dtype
            )  # 1, L, Feat

            if random.random() < audio_cfg_dropout:
                audio_cond = {"audio_emb": torch.zeros_like(audio_tensor)}
            else:
                audio_cond = {"audio_emb": audio_tensor}

            # Sample timestep per sample (CPU index for scheduler; GPU tensor for model)
            t_idx = torch.randint(
                low=0,
                high=pipe.scheduler.timesteps.shape[0],
                size=(1,),
                device=torch.device("cpu"),
            )
            timestep_cpu = pipe.scheduler.timesteps[t_idx]
            timestep_gpu = timestep_cpu.to(device=device, dtype=torch_dtype)
            noise = torch.randn_like(clean_latents)
            noised = pipe.scheduler.add_noise(clean_latents, noise, timestep_cpu)
            noised = noised.to(device=device, dtype=torch_dtype)
            if "context" in prompt_emb:
                prompt_emb["context"] = prompt_emb["context"].to(
                    device=device, dtype=torch_dtype
                )

            # Keep prefix latent frames from clean (simulate reference frame conditioning)
            noised[:, :, :prefix_lat] = clean_latents[:, :, :prefix_lat]

            # Predict and compute weighted MSE loss
            pred = model_for_step(
                noised, timestep=timestep_gpu, **prompt_emb, **image_emb, **audio_cond
            )
            target = pipe.scheduler.training_target(clean_latents, noise, timestep_cpu)
            target = target.to(device=noised.device, dtype=noised.dtype)
            weight = (
                pipe.scheduler.training_weight(timestep_cpu)
                .view(1, 1, 1, 1, 1)
                .to(device=noised.device, dtype=noised.dtype)
            )
            loss = (weight * (pred - target).pow(2)).mean()

            if use_deepspeed:
                model_for_step.backward(loss)
                model_for_step.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in dit.parameters() if p.requires_grad], 1.0
                )
                opt.step()

            if dist.get_rank() == 0:
                ema_loss = (
                    loss.item()
                    if ema_loss is None
                    else ema_loss * ema_decay + (1 - ema_decay) * loss.item()
                )
                pbar.set_description(
                    f"step {step} loss {loss.item():.4f} ema {ema_loss:.4f}"
                )
                pbar.update(1)
                # VRAM usage logging
                try:
                    alloc_gb = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                    max_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    try:
                        pbar.write(
                            f"[mem] alloc={alloc_gb:.2f}G reserved={reserved_gb:.2f}G max_alloc={max_alloc_gb:.2f}G"
                        )
                    except Exception:
                        print(
                            f"[mem] alloc={alloc_gb:.2f}G reserved={reserved_gb:.2f}G max_alloc={max_alloc_gb:.2f}G",
                            flush=True,
                        )
                except Exception:
                    pass
                # Periodic nvidia-smi
                if smi_every > 0 and (step % smi_every == 0):
                    try:
                        out = (
                            subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "--query-gpu=memory.total,memory.used",
                                    "--format=csv,noheader",
                                ]
                            )
                            .decode()
                            .strip()
                        )
                        try:
                            pbar.write(f"[nvsmi] {out}")
                        except Exception:
                            print(f"[nvsmi] {out}", flush=True)
                    except Exception:
                        pass

            step += 1
            if step % save_every == 0 and dist.get_rank() == 0:
                # Save LoRA adapter weights only
                ckpt_path = os.path.join(exp_dir, "pytorch_model.pt")
                module_to_save = model_for_step.module if use_deepspeed else dit
                torch.save(
                    {
                        k: v.to(torch.float32).cpu()
                        for k, v in module_to_save.state_dict().items()
                        if "lora" in k.lower()
                    },
                    ckpt_path,
                )
                with open(os.path.join(exp_dir, "config.json"), "w") as f:
                    json.dump(
                        {
                            "train_architecture": "lora",
                            "lora_rank": lora_rank,
                            "lora_alpha": lora_alpha,
                            "lora_target_modules": lora_target_modules,
                        },
                        f,
                    )

                # Evaluation on the held-out sample: save video + prompt + inputs
                try:
                    if holdout_video is not None:
                        # Pause training UI and announce eval start
                        try:
                            pbar.write(
                                f"[eval] Pausing training at step {step} (loss={loss.item():.4f}, ema={ema_loss:.4f})"
                            )
                        except Exception:
                            print(
                                f"[eval] Starting held-out eval at step {step} ...",
                                flush=True,
                            )
                        # Resolution pick
                        vid_frames, _, _ = read_video(holdout_video, pts_unit="sec")
                        Hh, Ww = vid_frames.shape[1], vid_frames.shape[2]
                        image_sizes = getattr(args, "image_sizes_720")
                        e_height, e_width = match_size(image_sizes, Hh, Ww)
                        # Latents and image embedding
                        eval_lat = load_or_build_video_latents(
                            cache_dir,
                            pipe,
                            holdout_video,
                            e_height,
                            e_width,
                            torch_dtype,
                        )
                        eval_img_lat = load_or_build_image_latent(
                            cache_dir, pipe, holdout_ref, e_height, e_width, torch_dtype
                        )
                        msk = torch.zeros_like(
                            eval_img_lat.repeat(1, 1, eval_lat.shape[2], 1, 1)[:, :1]
                        )
                        image_cat = eval_img_lat.repeat(1, 1, eval_lat.shape[2], 1, 1)
                        msk[:, :, 1:] = 1
                        image_emb_eval = {"y": torch.cat([image_cat, msk], dim=1)}
                        # Audio embedding (align to frames of eval_lat)
                        num_video_frames_eval = eval_lat.shape[2] * 4 - 3
                        audio_emb_eval, audio_len_frames_eval = (
                            prepare_audio_embeddings(
                                holdout_audio,
                                wav_feature_extractor,
                                audio_encoder,
                                sample_rate,
                                fps,
                                device,
                                torch_dtype,
                            )
                        )
                        first_fixed_frame = 1
                        need = num_video_frames_eval - first_fixed_frame
                        if audio_len_frames_eval < need:
                            audio_len_frames_eval = audio_len_frames_eval + (
                                need - audio_len_frames_eval % need
                            )
                        elif (audio_len_frames_eval - need) % (
                            num_video_frames_eval - 1
                        ) != 0:
                            audio_len_frames_eval = audio_len_frames_eval + (
                                (num_video_frames_eval - 1)
                                - (audio_len_frames_eval - need)
                                % (num_video_frames_eval - 1)
                            )
                        audio_emb_eval = F.pad(
                            audio_emb_eval,
                            (0, 0, 0, audio_len_frames_eval - audio_emb_eval.shape[0]),
                            mode="constant",
                            value=0,
                        )
                        audio_prefix = torch.zeros_like(
                            audio_emb_eval[:first_fixed_frame]
                        )
                        audio_tensor_eval = torch.cat(
                            [audio_prefix, audio_emb_eval[:need]], dim=0
                        )
                        audio_tensor_eval = audio_tensor_eval.unsqueeze(0).to(
                            device=device, dtype=torch_dtype
                        )
                        audio_cond_eval = {"audio_emb": audio_tensor_eval}

                        # Prompt embedding
                        pipe.load_models_to_device(["text_encoder"])
                        prompt_emb_eval = pipe.encode_prompt(
                            holdout_prompt, positive=True
                        )

                        # Compute eval loss on a single FlowMatch step (holdout) first
                        try:
                            pipe.scheduler.set_timesteps(
                                num_inference_steps=int(
                                    getattr(args, "num_train_timesteps", 1000)
                                ),
                                training=True,
                            )
                            # Optionally average eval loss over multiple random timesteps
                            eval_losses = []
                            for _s in range(max(1, eval_loss_samples)):
                                t_idx_eval = torch.randint(
                                    low=0,
                                    high=pipe.scheduler.timesteps.shape[0],
                                    size=(1,),
                                    device=torch.device("cpu"),
                                )
                                t_cpu_eval = pipe.scheduler.timesteps[t_idx_eval]
                                t_gpu_eval = t_cpu_eval.to(
                                    device=device, dtype=torch_dtype
                                )
                                noise_eval = torch.randn_like(eval_lat)
                                noised_eval = pipe.scheduler.add_noise(
                                    eval_lat, noise_eval, t_cpu_eval
                                ).to(device=device, dtype=torch_dtype)
                                pred_eval_loss = pipe.dit(
                                    noised_eval,
                                    timestep=t_gpu_eval,
                                    **prompt_emb_eval,
                                    **image_emb_eval,
                                    **audio_cond_eval,
                                )
                                target_eval = pipe.scheduler.training_target(
                                    eval_lat, noise_eval, t_cpu_eval
                                ).to(device=noised_eval.device, dtype=noised_eval.dtype)
                                weight_eval = (
                                    pipe.scheduler.training_weight(t_cpu_eval)
                                    .view(1, 1, 1, 1, 1)
                                    .to(
                                        device=noised_eval.device,
                                        dtype=noised_eval.dtype,
                                    )
                                )
                                loss_val = (
                                    (
                                        weight_eval
                                        * (pred_eval_loss - target_eval).pow(2)
                                    )
                                    .mean()
                                    .item()
                                )
                                eval_losses.append(loss_val)
                            eval_loss = float(sum(eval_losses) / len(eval_losses))
                            try:
                                pbar.write(
                                    f"[eval] Loss: {eval_loss:.6f} at step {step} (avg of {len(eval_losses)} t)"
                                )
                            except Exception:
                                print(
                                    f"[eval] Loss: {eval_loss:.6f} at step {step} (avg of {len(eval_losses)} t)",
                                    flush=True,
                                )
                        except Exception:
                            err = traceback.format_exc()
                            try:
                                pbar.write(
                                    "[eval] ERROR computing eval loss; skipping early-stop for this eval."
                                )
                                pbar.write(err)
                            except Exception:
                                print(
                                    "[eval] ERROR computing eval loss; skipping early-stop for this eval.",
                                    flush=True,
                                )
                                print(err, flush=True)
                            eval_loss = None

                        # Early stopping based on eval loss
                        if eval_loss is not None:
                            if (best_eval is None) or (
                                (best_eval - eval_loss) > early_stop_min_delta
                            ):
                                best_eval = eval_loss
                                best_eval_step = step
                                eval_no_improve = 0
                                try:
                                    pbar.write(
                                        f"[early-stop] New best eval {best_eval:.6f} at step {best_eval_step}"
                                    )
                                except Exception:
                                    print(
                                        f"[early-stop] New best eval {best_eval:.6f} at step {best_eval_step}",
                                        flush=True,
                                    )
                            else:
                                eval_no_improve += 1
                                step_delta = step - best_eval_step
                                debug_line = (
                                    f"[early-stop] no_improve_evals={eval_no_improve}, step_delta={step_delta}, "
                                    f"best_eval={best_eval:.6f}, curr_eval={eval_loss:.6f}, min_delta={early_stop_min_delta}, "
                                    f"patience_steps={early_stop_patience}, patience_evals={early_stop_patience_evals}"
                                )
                                try:
                                    pbar.write(debug_line)
                                except Exception:
                                    print(debug_line, flush=True)
                                if step_delta >= early_stop_patience or (
                                    early_stop_patience_evals > 0
                                    and eval_no_improve >= early_stop_patience_evals
                                ):
                                    msg = (
                                        f"[early-stop] No eval improvement for {step_delta} steps"
                                        f" ({eval_no_improve} evals). Best {best_eval:.6f} at {best_eval_step}. Stopping."
                                    )
                                    try:
                                        pbar.write(msg)
                                    except Exception:
                                        print(msg, flush=True)
                                    stop_training = True
                                    break

                        if stop_training:
                            break

                        # Short denoise preview using pipeline's scheduler (only if not stopping)
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        preview_steps = max(1, int(preview_timesteps_eval))
                        pipe.scheduler.set_timesteps(
                            num_inference_steps=preview_steps, training=False
                        )
                        latents_eval = torch.randn_like(eval_lat).to(
                            device=pipe.device, dtype=torch_dtype
                        )
                        pipe.load_models_to_device(["dit"])
                        prev_training_mode = pipe.dit.training
                        pipe.dit.eval()
                        with torch.no_grad():
                            for progress_id, timestep in enumerate(
                                pipe.scheduler.timesteps
                            ):
                                timestep = timestep.unsqueeze(0).to(
                                    dtype=torch_dtype, device=pipe.device
                                )
                                pred_eval = pipe.dit(
                                    latents_eval,
                                    timestep=timestep,
                                    **prompt_emb_eval,
                                    **image_emb_eval,
                                    **audio_cond_eval,
                                )
                                latents_eval = pipe.scheduler.step(
                                    pred_eval,
                                    pipe.scheduler.timesteps[progress_id],
                                    latents_eval,
                                )
                        pipe.dit.train(prev_training_mode)
                        try:
                            pipe.load_models_to_device(["vae"])
                            with torch.no_grad():
                                frames_eval = pipe.decode_video(latents_eval)
                        except Exception:
                            err = traceback.format_exc()
                            try:
                                pbar.write(
                                    "[eval] ERROR during preview/decoding; skipping video save."
                                )
                                pbar.write(err)
                            except Exception:
                                print(
                                    "[eval] ERROR during preview/decoding; skipping video save.",
                                    flush=True,
                                )
                                print(err, flush=True)
                            frames_eval = None
                        frames_eval = (
                            frames_eval[0].permute(1, 2, 3, 0).float() + 1
                        ) / 2  # T H W C in [0,1]
                        eval_dir = os.path.join(exp_dir, "eval")
                        os.makedirs(eval_dir, exist_ok=True)
                        # Eval stats
                        try:
                            T_eval, H_eval, W_eval, _ = frames_eval.shape
                            secs = T_eval / max(1, fps)
                            try:
                                pbar.write(
                                    f"[eval] Stats: frames={T_eval}, res={H_eval}x{W_eval}, fps={fps}, secs={secs:.2f}, ema={ema_loss:.4f}"
                                )
                            except Exception:
                                print(
                                    f"[eval] Stats: frames={T_eval}, res={H_eval}x{W_eval}, fps={fps}, secs={secs:.2f}, ema={ema_loss:.4f}",
                                    flush=True,
                                )
                        except Exception:
                            pass
                        out_mp4 = os.path.join(eval_dir, f"step_{step:06d}.mp4")
                        out_size = -1
                        if frames_eval is not None:
                            try:
                                save_tensor_video_as_mp4(out_mp4, frames_eval, fps)
                                out_size = (
                                    os.path.getsize(out_mp4)
                                    if os.path.exists(out_mp4)
                                    else -1
                                )
                            except Exception:
                                out_size = -1
                                err = traceback.format_exc()
                                try:
                                    pbar.write(
                                        "[eval] ERROR saving local MP4; skipping volume copy."
                                    )
                                    pbar.write(err)
                                except Exception:
                                    print(
                                        "[eval] ERROR saving local MP4; skipping volume copy.",
                                        flush=True,
                                    )
                                    print(err, flush=True)
                        try:
                            pbar.write(
                                f"[eval] Saved eval video to {out_mp4} (size={out_size} bytes)"
                            )
                        except Exception:
                            print(
                                f"[eval] Saved eval video to {out_mp4} (size={out_size} bytes)",
                                flush=True,
                            )
                        # Save a few debug frames (first/middle/last) and basic stats
                        try:
                            if frames_eval is not None:
                                T_eval = frames_eval.shape[0]

                                def _save_frame(idx, tag):
                                    fr = (
                                        (frames_eval[idx] * 255.0)
                                        .clamp(0, 255)
                                        .to(torch.uint8)
                                        .cpu()
                                        .numpy()
                                    )
                                    img = Image.fromarray(fr)
                                    img.save(
                                        os.path.join(
                                            eval_dir, f"step_{step:06d}_{tag}.jpg"
                                        ),
                                        quality=95,
                                    )

                                _save_frame(0, "f0")
                                _save_frame(min(T_eval // 2, T_eval - 1), "fmid")
                                _save_frame(T_eval - 1, "flast")
                                fmin = float(frames_eval.min().item())
                                fmax = float(frames_eval.max().item())
                                fmean = float(frames_eval.mean().item())
                                fstd = float(frames_eval.std().item())
                                try:
                                    pbar.write(
                                        f"[eval] frame_stats min={fmin:.3f} max={fmax:.3f} mean={fmean:.3f} std={fstd:.3f}; preview_steps={preview_steps}"
                                    )
                                except Exception:
                                    print(
                                        f"[eval] frame_stats min={fmin:.3f} max={fmax:.3f} mean={fmean:.3f} std={fstd:.3f}; preview_steps={preview_steps}",
                                        flush=True,
                                    )
                        except Exception:
                            pass

                        # Copy to Modal Volume if available for easy fetch
                        try:
                            vol_eval_dir = os.path.join("/vol", "outputs", "eval")
                            os.makedirs(vol_eval_dir, exist_ok=True)
                            # Debug: report writability
                            try:
                                can_write = os.access(vol_eval_dir, os.W_OK)
                                pbar.write(
                                    f"[eval] Volume dir {vol_eval_dir} writeable={can_write}"
                                )
                            except Exception:
                                pass
                            vol_mp4 = os.path.join(vol_eval_dir, f"step_{step:06d}.mp4")
                            if os.path.exists(out_mp4) and out_size > 0:
                                shutil.copyfile(out_mp4, vol_mp4)
                            vol_size = (
                                os.path.getsize(vol_mp4)
                                if os.path.exists(vol_mp4)
                                else -1
                            )
                            vol_name = os.environ.get(
                                "OMNIAVATAR_MODAL_VOLUME", "omniavatar-cache"
                            )
                            fetch_cmd = f"modal volume get {vol_name} outputs/eval/step_{step:06d}.mp4 ./step_{step:06d}.mp4"
                            try:
                                pbar.write(
                                    f"[eval] Also saved to volume: {vol_mp4} (size={vol_size} bytes)"
                                )
                                pbar.write(f"[eval] Fetch locally with: {fetch_cmd}")
                            except Exception:
                                print(
                                    f"[eval] Also saved to volume: {vol_mp4} (size={vol_size} bytes)",
                                    flush=True,
                                )
                                print(
                                    f"[eval] Fetch locally with: {fetch_cmd}",
                                    flush=True,
                                )
                        except Exception:
                            err = traceback.format_exc()
                            try:
                                pbar.write(
                                    "[eval] Warning: Modal volume not mounted or copy failed; skipping volume copy."
                                )
                                pbar.write(err)
                            except Exception:
                                print(
                                    "[eval] Warning: Modal volume not mounted or copy failed; skipping volume copy.",
                                    flush=True,
                                )
                                print(err, flush=True)
                        try:
                            pbar.write("[eval] Resuming training...")
                        except Exception:
                            print("[eval] Resuming training...", flush=True)
                        with open(
                            os.path.join(eval_dir, f"step_{step:06d}.txt"), "w"
                        ) as ftxt:
                            ftxt.write(holdout_prompt.strip() + "\n")
                        try:
                            Image.open(holdout_ref).save(
                                os.path.join(eval_dir, f"step_{step:06d}_ref.jpg")
                            )
                        except Exception:
                            pass
                        try:
                            shutil.copyfile(
                                holdout_audio,
                                os.path.join(eval_dir, f"step_{step:06d}.wav"),
                            )
                        except Exception:
                            pass
                except Exception:
                    err = traceback.format_exc()
                    try:
                        pbar.write("[eval] FATAL error in eval block:")
                        pbar.write(err)
                    except Exception:
                        print("[eval] FATAL error in eval block:", flush=True)
                        print(err, flush=True)

            # Early stopping now handled after eval based on eval_loss

    if dist.get_rank() == 0:
        ckpt_path = os.path.join(exp_dir, "pytorch_model.pt")
        module_to_save = model_for_step.module if use_deepspeed else dit
        torch.save(
            {
                k: v.to(torch.float32).cpu()
                for k, v in module_to_save.state_dict().items()
                if "lora" in k.lower()
            },
            ckpt_path,
        )
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "train_architecture": "lora",
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "lora_target_modules": lora_target_modules,
                },
                f,
            )


if __name__ == "__main__":
    main()
