from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from huggingface_hub import snapshot_download

try:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "缺少本地推理依赖，请先安装 transformers、accelerate、huggingface-hub 和 torch。"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model"
DEFAULT_REPO_ID = "Qwen/Qwen3.5-4B"
DEFAULT_PROMPT = "请用中文简要说明数据库系统的组成，并分点回答。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and test local Qwen inference")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for local generation test")
    parser.add_argument(
        "--max-new-tokens",
        "--max-tokens",
        dest="max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens",
    )
    parser.add_argument("--download-only", action="store_true", help="Only download weights without generation")
    parser.add_argument("--image-url", help="Optional remote image URL for multimodal inference")
    parser.add_argument("--image-path", type=Path, help="Optional local image path for multimodal inference")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode. By default the script runs in non-thinking mode.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=1.5,
        help="Presence penalty applied during generation",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Preferred device for local inference",
    )
    args = parser.parse_args()
    ensure_runtime_support()

    local_model_dir = get_model_dir(args.repo_id)
    print(f"[local-qwen] 模型仓库: {args.repo_id}")
    print(f"[local-qwen] 本地目录: {local_model_dir}")

    download_model(args.repo_id, local_model_dir)

    if args.download_only:
        print("[local-qwen] 下载完成，已跳过推理测试。")
        return

    result = run_generation_test(
        model_dir=local_model_dir,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        image_url=args.image_url,
        image_path=args.image_path,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def get_model_dir(repo_id: str) -> Path:
    safe_name = repo_id.split("/")[-1]
    return MODEL_ROOT / safe_name


def download_model(repo_id: str, local_model_dir: Path) -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    local_model_dir.mkdir(parents=True, exist_ok=True)
    print("[local-qwen] 开始下载或校验模型文件...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[local-qwen] 模型文件已准备完成。")


def ensure_runtime_support() -> None:
    if AutoModelForImageTextToText is not None:
        return
    raise SystemExit(
        "当前环境中的 transformers 版本过旧，无法本地加载 Qwen3.5。"
        '请升级到最新主分支版本，例如: pip install -U "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"'
    )


def run_generation_test(
    model_dir: Path,
    prompt: str,
    max_new_tokens: int,
    device: str,
    image_url: str | None,
    image_path: Path | None,
    enable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
) -> dict[str, object]:
    print("[local-qwen] 开始加载 processor...")
    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer

    print("[local-qwen] 开始加载模型，这一步可能较慢...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=get_torch_dtype(device),
        device_map=get_device_map(device),
    )
    print("[local-qwen] 模型加载完成。")

    messages = build_hf_messages(
        prompt=prompt,
        image_url=image_url,
        image_path=image_path,
    )
    print("[local-qwen] 开始构造输入...")
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    images = load_images(image_url=image_url, image_path=image_path)
    if images:
        model_inputs = processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "pad_token_id": tokenizer.eos_token_id,
        }
    else:
        model_inputs = tokenizer([prompt_text], return_tensors="pt")
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    model_inputs = model_inputs.to(model.device)

    print("[local-qwen] 开始本地推理...")
    started_at = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **generation_kwargs,
        )
    elapsed = round(time.time() - started_at, 2)

    input_len = model_inputs.input_ids.shape[1]
    output_ids = generated_ids[:, input_len:]
    answer = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "model_dir": str(model_dir),
        "device": str(model.device),
        "prompt": prompt,
        "image_url": image_url,
        "image_path": str(image_path.resolve()) if image_path else None,
        "enable_thinking": enable_thinking,
        "presence_penalty": presence_penalty,
        "transformers_version": transformers.__version__,
        "elapsed_seconds": elapsed,
        "answer": answer,
    }


def build_hf_messages(prompt: str, image_url: str | None, image_path: Path | None) -> list[dict[str, object]]:
    content: list[dict[str, str]] = []
    if image_url:
        content.append({"type": "image", "image": image_url})
    if image_path:
        resolved_path = image_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {resolved_path}")
        content.append({"type": "image", "image": str(resolved_path)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_images(image_url: str | None, image_path: Path | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    if image_url:
        with urlopen(image_url) as response:
            images.append(Image.open(BytesIO(response.read())).convert("RGB"))
    if image_path:
        resolved_path = image_path.expanduser().resolve()
        images.append(Image.open(resolved_path).convert("RGB"))
    return images


def get_device_map(device: str) -> str | None:
    if device == "cpu":
        return None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境未检测到 CUDA，不能使用 --device cuda。")
        return "auto"
    return "auto" if torch.cuda.is_available() else None


def get_torch_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


if __name__ == "__main__":
    main()
