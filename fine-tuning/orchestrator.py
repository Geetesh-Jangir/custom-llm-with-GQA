import argparse, shlex, subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent


# def run(cmd: str):
#     print(f"\n>>> {cmd}")
#     res = subprocess.run(shlex.split(cmd), cwd=ROOT, check=True)
#     if res.returncode != 0:
#         sys.exit(res.returncode)


def run(cmd: str):

    # cmd_list = [sys.executable] + shlex.split(cmd)

    print(f"\n>>> Running:\n{cmd}\n")

    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        run(
            "python train_sft.py --data huggingface --ckpt ../base-model/runs/part-demo/model_last.pt --out runs/sft-demo --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128"
        )
        run(
            'python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt "What are the three primary colors?" --tokens 30 --temperature 0.2'
        )
        # run(
        #     'python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt "What does DNA stand for?" --tokens 30 --temperature 0.2'
        # )
        # run(
        #     'python sample_sft.py --ckpt runs/sft-demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt "Reverse engineer this code to create a new version\ndef factorialize(num):\n  factorial = 1\n  for i in range(1, num):\n    factorial *= i\n  \n  return factorial" --tokens 64 --temperature 0.2'
        # )

    print("\ncode runned successfully")
