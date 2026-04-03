from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Any, Dict


class NoopLogger:
    def log(self, **kwargs):
        pass

    def close(
        self,
    ):
        pass


class TBLogger(NoopLogger):
    def __init__(self, out_dir: str, flush_secs: int = 10, run_name: str | None = None):
        self.w = None
        self.hparams_logged = False
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(out_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.w = SummaryWriter(log_dir=str(run_dir), flush_secs=flush_secs)
        except Exception as e:
            print(f"[TBLogger] TensorBoard not available: {e}. Logging disabled")
        self._auto_hist_max_elems = 2048
        self.run_dir = str(run_dir)

    def log(self, step: Optional[int] = None, **kv: Any):
        if not self.w:
            return
        for k, v in kv.items():
            # key channel option via key preffix
            if isinstance(k, str) and k.startswith("text/"):
                try:
                    self.w.add_text(k[5:], str(v), global_step=step)
                except Exception:
                    pass
                continue

            try:
                import torch, numpy as np

                is_torch = isinstance(v, torch.Tensor)
                is_numpy = isinstance(v, np.ndarray)
                if is_torch or is_numpy:
                    # if there is only one element in v then it is a scalar otherwise it is need to be ploted as histograph
                    numel = int(v.numel() if is_torch else v.size)
                    if numel == 1:
                        val = v.item() if is_torch else float(v)
                        self.w.add_scalar(k, float(val), global_step=step)
                    else:
                        if numel <= self._auto_hist_max_elems:
                            self.w.add_histogram(
                                k, v.detach().cpu() if is_torch else v, global_step=step
                            )
                        else:
                            # fall back to scalar if num of elem is bigger
                            arr = (
                                v.detach().cpu().flatten().numpy()
                                if is_torch
                                else v.flatten()
                            )
                            self.w.add_scalar(
                                k + "/mean", float(arr.mean()), global_step=step
                            )
                            self.w.add_scalar(
                                k + "/std", float(arr.std()), global_step=step
                            )
                    continue
            except Exception:
                pass

            # if passed only a number
            try:
                self.w.add_scalar(k, float(v), global_step=step)
            except Exception:
                pass

    # creating helper function for all
    def hist(
        self,
        tag: str,
        values: Any,
        step: Optional[int] = None,
        bins: str = "tensorflow",
    ):
        if not self.w:
            return
        try:
            import torch

            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            self.w.add_histogram(tag, values, global_step=step, bins=bins)
        except Exception:
            pass

    def text(self, tag: str, text: str, step: Optional[int] = None):
        if not self.w:
            return
        try:
            self.w.add_text(tag, text, global_step=step)
        except Exception:
            pass

    def image(self, tag: str, img, step: Optional[int] = None):
        """
        img:torch.tensor [C,H,W] or [H,W,C] if img: numpy array
        """
        if not self.w:
            return
        try:
            self.w.add_image(
                tag,
                img,
                global_step=step,
                dataformats=(
                    "CHW"
                    if getattr(img, "ndim", 0) == 3 and img.shape[0] in (1, 3)
                    else "HWC"
                ),
            )
        except Exception:
            pass

    def graph(self, model, example_input):
        if not self.w:
            return
        try:
            if not isinstance(example_input, tuple):
                example_input = (example_input,)
            self.w.add_graph(model, example_input)
        except Exception:
            pass

    def hparams(
        self, hparams: Dict[str, Any], metrics_once: Optional[Dict[str, float]] = None
    ):
        if not self.w or self.hparams_logged:
            return
        try:
            self.w.add_hparams(hparams, metrics_once or {}, run_name="_hparams")
            self.hparams_logged = True
        except Exception:
            pass

    def flush(
        self,
    ):
        if self.w:
            try:
                self.w.flush()
            except Exception:
                pass

    def close(
        self,
    ):
        if self.w:
            try:
                self.w.close()
            except Exception:
                pass


class WBLogger(NoopLogger):
    def __init__(self, project: str, run_name: str | None = None):
        try:
            import wandb

            wandb.init(project=project, name=run_name)
            self.wb = wandb
        except Exception:
            self.wb = None

    def log(self, **kv):
        if self.wb:
            self.wb.log(kv)


def init_logger(which: str, out_dir: str = "runs/part4"):
    if which == "tensorboard":
        tb = TBLogger(out_dir)
        return tb if tb.w is not None else NoopLogger()
    if which == "wandb":
        return WBLogger(project="llm-part4")
    return NoopLogger()
