import argparse, pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve().parent


def run(cmd: str):
    print(f"\n>>> {cmd}")
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        run(
            "python train.py --data tiny.txt --out runs/part-demo --bpe --vocab_size 8000 --epochs 1 --steps 500 --batch_size 16 --block_size 128 --n_layer 2 --n_head 2 --n_embd 128 --mixed_precision --grad_accum_steps 2 --log tensorboard"
        )
        run(
            'python sample.py --ckpt runs/part-demo/model_last.pt --tokens 100 --prompt "Generate a short story"'
        )
