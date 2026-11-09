"""
Inference with multi-view test-time augmentation (TTA) for a trained <=5M-parameter model.

Each test image is classified from 18 views: horizontal flip x 9 small shifts
(reflect-pad by --shift pixels, crop at offsets in {-s, 0, +s}^2). Softmax outputs are averaged.
Only the single trained model is used; no test-set statistics are computed or used.

  python predict.py --model runs/full_e400_color_cutmix/model.pt --out submission_tta.csv
"""
import argparse
import contextlib
import csv

import numpy as np
import torch
import torch.nn.functional as F

from train import GPUPrep, PARAM_LIMIT, build_model, count_params, load_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model.pt written by train.py")
    p.add_argument("--data", default="data.zip")
    p.add_argument("--cache", default="cifar_cache.npz")
    p.add_argument("--out", default="submission_tta.csv")
    p.add_argument("--shift", type=int, default=2, help="pixel shift for TTA crops (0 = flip-only)")
    p.add_argument("--bs", type=int, default=500)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    model = build_model(ck["config"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    n = count_params(model)
    print(f"trainable parameters: {n:,}  (limit {PARAM_LIMIT:,})")
    assert n <= PARAM_LIMIT, "model exceeds the 5M parameter budget"
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16)
    else:
        amp = contextlib.nullcontext

    _, _, x_test, test_ids = load_data(a.data, a.cache)
    prep = GPUPrep(dev, channels_last=dev.type == "cuda")
    s = a.shift
    offsets = [(dy, dx) for dy in (-s, 0, s) for dx in (-s, 0, s)] if s > 0 else [(0, 0)]

    x = torch.from_numpy(x_test).permute(0, 3, 1, 2).contiguous()
    p_tta, p_flip = [], []
    with torch.no_grad():
        for i in range(0, len(x), a.bs):
            xb = prep(x[i:i + a.bs])
            padded = F.pad(xb, (s, s, s, s), mode="reflect") if s > 0 else xb
            acc, base = 0, 0
            for dy, dx in offsets:
                v = padded[:, :, s + dy:s + dy + 32, s + dx:s + dx + 32]
                for flip in (False, True):
                    with amp():
                        pr = F.softmax(model(torch.flip(v, dims=[3]) if flip else v).float(), 1)
                    acc = acc + pr
                    if (dy, dx) == (0, 0):
                        base = base + pr
            p_tta.append((acc / (2 * len(offsets))).cpu())
            p_flip.append((base / 2).cpu())
    pred = torch.cat(p_tta).argmax(1).numpy()
    pred_flip = torch.cat(p_flip).argmax(1).numpy()

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(zip(test_ids.tolist(), pred.tolist()))
    print(f"views per image: {2 * len(offsets)}")
    print(f"agreement with flip-only predictions: {(pred == pred_flip).mean():.4f} "
          f"({int((pred != pred_flip).sum())} images changed)")
    print("predicted class counts:", np.bincount(pred, minlength=10).tolist())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
