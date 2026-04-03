from __future__ import annotations
import argparse, torch

# reuse gpt modern
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "base-model"))
from model_modern import GPTModern

from collator_sft import SFTCollator
from formatters import format_prompt_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--bpe_dir", type=str, default="../base-model/runs/part-demo/tokenizer"
    )
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})

    col = SFTCollator(block_size=cfg.get("block_size", 256), bpe_dir=args.bpe_dir)
    model = GPTModern(
        vocab_size=col.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        use_rmsnorm=True,
        use_swiglu=True,
        rope=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt_text = format_prompt_only(args.prompt).replace("</s>", "")
    ids = col.encode(prompt_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=3
        )

    out_ids = out[0].tolist()
    orig_len = idx.size(1)

    if hasattr(col, "tok") and hasattr(col.tok, "decode"):
        generated = col.tok.decode(out_ids)
        print(generated)
    else:
        generated = bytes(out_ids[orig_len:]).decode("utf-8", errors="ignore")
        print(generated)


if __name__ == "__main__":
    main()
