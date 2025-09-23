import hashlib
import json
import math
import os
import random
import sys
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
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
    input_values = torch.from_numpy(input_values).float().to(device=device)
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

    # Inject LoRA into DiT and select trainable params
    assert (
        getattr(args, "train_architecture", "lora") == "lora"
    ), "This script currently supports LoRA fine-tuning only."
    lora_rank = int(getattr(args, "lora_rank", 128))
    lora_alpha = int(getattr(args, "lora_alpha", 64))
    lora_target_modules = getattr(args, "lora_target_modules", "q,k,v,o,ffn.0,ffn.2")
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True,
        target_modules=lora_target_modules.split(","),
    )
    dit = inject_adapter_in_model(lora_cfg, pipe.denoising_model())

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

    # Audio encoder (frozen)
    wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    audio_encoder = Wav2VecModel.from_pretrained(
        args.wav2vec_path, local_files_only=True
    ).to(device=device)
    audio_encoder.feature_extractor._freeze_parameters()
    for p in audio_encoder.parameters():
        p.requires_grad_(False)

    # Data
    train_dataset = ManifestDataset(getattr(args, "train_manifest"))
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
    opt = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )

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

    total_steps = int(getattr(args, "num_train_steps", 10000))
    save_every = int(getattr(args, "save_every", 1000))

    exp_dir = (
        args.exp_path
        if os.path.isabs(args.exp_path)
        else os.path.join("checkpoints", args.exp_path)
    )
    os.makedirs(exp_dir, exist_ok=True)

    step = 0
    dit.train()
    pbar = tqdm(total=total_steps, disable=dist.get_rank() != 0)

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break
            prompt_list = batch["prompt"]
            ref_img_list = batch["ref_img"]
            audio_list = batch["audio"]
            video_list = batch["video"]

            # For simplicity, operate per-sample (effective batch size 1). Accumulation can be added later.
            opt.zero_grad(set_to_none=True)

            # Prepare prompt
            prompt = prompt_list[0]
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
            num_video_frames = int(max_tokens * 16 * 16 * 4 / height / width)
            num_video_frames = (
                num_video_frames // 4 * 4 + 1
                if num_video_frames % 4 != 0
                else num_video_frames - 3
            )
            num_latent_frames = (num_video_frames + 3) // 4
            # Trim/pad clean_latents to T_lat
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

            # Sample timestep per sample
            t_idx = torch.randint(
                low=0, high=pipe.scheduler.timesteps.shape[0], size=(1,), device=device
            )
            timestep = pipe.scheduler.timesteps[t_idx]
            noise = torch.randn_like(clean_latents)
            noised = pipe.scheduler.add_noise(clean_latents, noise, timestep)

            # Keep prefix latent frames from clean (simulate reference frame conditioning)
            noised[:, :, :prefix_lat] = clean_latents[:, :, :prefix_lat]

            # Predict and compute weighted MSE loss
            pred = pipe.dit(
                noised, timestep=timestep, **prompt_emb, **image_emb, **audio_cond
            )
            target = pipe.scheduler.training_target(clean_latents, noise, timestep)
            weight = (
                pipe.scheduler.training_weight(timestep)
                .view(1, 1, 1, 1, 1)
                .to(device=noised.device, dtype=noised.dtype)
            )
            loss = (weight * (pred - target).pow(2)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in dit.parameters() if p.requires_grad], 1.0
            )
            opt.step()

            if dist.get_rank() == 0:
                pbar.set_description(f"step {step} loss {loss.item():.4f}")
                pbar.update(1)

            step += 1
            if step % save_every == 0 and dist.get_rank() == 0:
                # Save LoRA adapter weights only
                ckpt_path = os.path.join(exp_dir, "pytorch_model.pt")
                torch.save(
                    {
                        k: v.to(torch.float32).cpu()
                        for k, v in dit.state_dict().items()
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

    if dist.get_rank() == 0:
        ckpt_path = os.path.join(exp_dir, "pytorch_model.pt")
        torch.save(
            {
                k: v.to(torch.float32).cpu()
                for k, v in dit.state_dict().items()
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
