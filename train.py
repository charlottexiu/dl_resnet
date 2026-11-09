"""
NYU Tandon Deep Learning F26 - Efficient ResNet Challenge (CIFAR-10, <= 5M params)

Single-file, reproducible pipeline:
  data.zip -> cached numpy arrays -> train ResNet from scratch -> val metrics -> submission.csv

Typical usage
  # 0) sanity check (runs in ~1 min, even on CPU)
  python train.py --data data.zip --smoke
  # 1) recipe check with a 5k held-out validation split
  python train.py --data data.zip --val_size 5000 --epochs 200 --out runs/val
  # 2) final model on all 50k training images -> submission.csv
  python train.py --data data.zip --val_size 0 --epochs 200 --out runs/full

Outputs in --out: submission.csv, model.pt, history.csv, curves.png, config.json, summary.json
"""
import argparse
import copy
import csv
import io
import json
import math
import os
import random
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PARAM_LIMIT = 5_000_000
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
# CIFAR-10 training-set channel statistics
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------- data
def _find_prefix(names, target):
    for n in names:
        if n.endswith(target) and "__MACOSX" not in n:
            return n[: -len(target)]
    raise FileNotFoundError(f"{target} not found inside the zip")


def load_data(zip_path, cache_path):
    """Decode all PNGs once and cache them as uint8 arrays (N, 32, 32, 3)."""
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["x_train"], d["y_train"], d["x_test"], d["test_ids"]

    t0 = time.time()
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        pre = _find_prefix(names, "train_labels.csv")

        def read_csv(name):
            with z.open(pre + name) as f:
                return list(csv.DictReader(io.TextIOWrapper(f, "utf-8")))

        def read_img(folder, fn):
            img = Image.open(io.BytesIO(z.read(f"{pre}{folder}/{fn}"))).convert("RGB")
            return np.asarray(img, dtype=np.uint8)

        tr = read_csv("train_labels.csv")
        te = read_csv("test.csv")
        x_train = np.stack([read_img("train_images", r["filename"]) for r in tr])
        y_train = np.array([int(r["label"]) for r in tr], dtype=np.int64)
        x_test = np.stack([read_img("test_images", r["filename"]) for r in te])
        test_ids = np.array([int(r["id"]) for r in te], dtype=np.int64)

    assert x_train.shape[1:] == (32, 32, 3), x_train.shape
    print(f"decoded {len(x_train)} train / {len(x_test)} test images in {time.time() - t0:.0f}s")
    if cache_path:
        np.savez(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, test_ids=test_ids)
    return x_train, y_train, x_test, test_ids


def stratified_split(y, val_size, seed):
    """Deterministic, class-balanced train/val split."""
    if val_size <= 0:
        return np.arange(len(y)), np.array([], dtype=np.int64)
    rng = np.random.RandomState(seed)
    per_class = val_size // 10
    tr_idx, va_idx = [], []
    for c in range(10):
        idx = rng.permutation(np.where(y == c)[0])
        va_idx.append(idx[:per_class])
        tr_idx.append(idx[per_class:])
    return np.concatenate(tr_idx), np.concatenate(va_idx)


class AugDataset(torch.utils.data.Dataset):
    """CPU-side augmentation on uint8 CHW tensors (normalisation/cutout happen on the GPU)."""

    def __init__(self, x_uint8_nhwc, y, augment, aug_policy, gray_p=0.0, jitter=0.0):
        self.x = torch.from_numpy(x_uint8_nhwc).permute(0, 3, 1, 2).contiguous()
        self.y = torch.from_numpy(y).long()
        self.tf = None
        if augment:
            import torchvision.transforms.v2 as T
            ops = [T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip()]
            if aug_policy == "trivial":
                ops.append(T.TrivialAugmentWide())
            elif aug_policy == "autoaugment":
                ops.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
            elif aug_policy == "randaugment":
                ops.append(T.RandAugment(num_ops=2, magnitude=9))
            if jitter > 0:  # test set: brighter backgrounds, lower saturation
                ops.append(T.RandomApply([T.ColorJitter(jitter, jitter, jitter)], p=0.5))
            if gray_p > 0:  # test set has ~3.7% grayscale images vs ~1% in train
                ops.append(T.RandomGrayscale(p=gray_p))
            self.tf = T.Compose(ops)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.tf is not None:
            x = self.tf(x)
        return x, self.y[i]


class GPUPrep:
    """uint8 -> normalised float on device, plus batched Cutout."""

    def __init__(self, device, channels_last):
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255
        self.device = device
        self.mf = torch.channels_last if channels_last else torch.contiguous_format

    def __call__(self, x):
        x = x.to(self.device, non_blocking=True).float()
        x = (x - self.mean) / self.std
        return x.contiguous(memory_format=self.mf)


def cutout(x, size):
    B, _, H, W = x.shape
    cy = torch.randint(0, H, (B, 1, 1), device=x.device)
    cx = torch.randint(0, W, (B, 1, 1), device=x.device)
    ys = torch.arange(H, device=x.device).view(1, H, 1)
    xs = torch.arange(W, device=x.device).view(1, 1, W)
    h = size // 2
    m = (ys >= cy - h) & (ys < cy + h) & (xs >= cx - h) & (xs < cx + h)
    return x * (~m).unsqueeze(1).to(x.dtype)


def cutmix(x, y, alpha=1.0):
    B, _, H, W = x.shape
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(B, device=x.device)
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
    x1, x2 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[idx], lam


# ----------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    """out = ReLU( F(x) + S(x) ),  F = conv-BN-ReLU-conv-BN,  S = identity or 1x1 conv-BN projection."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        nn.init.zeros_(self.bn2.weight)  # zero-init residual: each block starts as identity
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNetCIFAR(nn.Module):
    def __init__(self, widths=(64, 128, 256, 368), blocks=(2, 2, 2, 1), stem=64, dropout=0.0, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        layers, c = [], stem
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                layers.append(BasicBlock(c, w, 2 if (j == 0 and i > 0) else 1))
                c = w
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(self.drop(torch.flatten(self.pool(x), 1)))


class WideBlock(nn.Module):
    """Pre-activation wide residual block (used only for the large KD teacher)."""

    def __init__(self, cin, cout, stride, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.dropout = dropout
        self.shortcut = None if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = x if self.shortcut is None else self.shortcut(o)
        o = F.relu(self.bn2(self.conv1(o)))
        if self.dropout > 0:
            o = F.dropout(o, self.dropout, self.training)
        return self.conv2(o) + sc


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, dropout=0.0, num_classes=10):
        super().__init__()
        n = (depth - 4) // 6
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        layers, c = [], 16
        for i, w in enumerate((16 * widen, 32 * widen, 64 * widen)):
            for j in range(n):
                layers.append(WideBlock(c, w, (1 if i == 0 else 2) if j == 0 else 1, dropout))
                c = w
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(c)
        self.fc = nn.Linear(c, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.layers(self.conv(x))))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def build_model(cfg):
    """cfg: dict of training args (also stored inside every model.pt)."""
    if cfg.get("arch", "resnet") == "wrn":
        return WideResNet(cfg.get("depth", 28), cfg.get("widen", 10), cfg.get("dropout", 0.0))
    return ResNetCIFAR(tuple(cfg["widths"]), tuple(cfg["blocks"]), cfg["stem"], cfg.get("dropout", 0.0))


def load_teacher(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    t = build_model(ck["config"]).to(device)
    t.load_state_dict(ck["state_dict"])
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    if device.type == "cuda":
        t = t.to(memory_format=torch.channels_last)
    return t


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict_probs(model, x_uint8, prep, bs, amp_ctx, tta):
    model.eval()
    ds = AugDataset(x_uint8, np.zeros(len(x_uint8), dtype=np.int64), augment=False, aug_policy="none")
    out = []
    for xb, _ in torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False):
        xb = prep(xb)
        with amp_ctx():
            p = F.softmax(model(xb).float(), 1)
            if tta:
                p = (p + F.softmax(model(torch.flip(xb, dims=[3])).float(), 1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- main
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data.zip")
    p.add_argument("--cache", default="cifar_cache.npz")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_size", type=int, default=5000, help="0 = train on all 50k")
    # model
    p.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 368])
    p.add_argument("--blocks", type=int, nargs="+", default=[2, 2, 2, 1])
    p.add_argument("--stem", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--arch", default="resnet", choices=["resnet", "wrn"], help="wrn = large KD teacher only")
    p.add_argument("--depth", type=int, default=28, help="WRN depth")
    p.add_argument("--widen", type=int, default=10, help="WRN widen factor")
    p.add_argument("--teacher_only", action="store_true",
                   help="allow > 5M params; writes model.pt for distillation but NO submission")
    # knowledge distillation
    p.add_argument("--teachers", nargs="*", default=[], help="model.pt files whose averaged predictions the student matches")
    p.add_argument("--kd_alpha", type=float, default=0.9, help="weight of the KD loss (rest = hard-label loss)")
    p.add_argument("--kd_T", type=float, default=2.0, help="KD temperature")
    # optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.2, help="peak LR for SGD at --bs")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup", type=int, default=5, help="warm-up epochs")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999, help="0 disables EMA")
    # augmentation
    p.add_argument("--aug", default="trivial", choices=["none", "trivial", "autoaugment", "randaugment"])
    p.add_argument("--cutout", type=int, default=16, help="0 disables")
    p.add_argument("--cutmix", type=float, default=0.0, help="probability of applying CutMix to a batch")
    p.add_argument("--gray", type=float, default=0.0, help="RandomGrayscale probability, e.g. 0.1")
    p.add_argument("--jitter", type=float, default=0.0, help="ColorJitter strength (brightness/contrast/saturation), e.g. 0.3")
    # system
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end to end")
    return p.parse_args()


def main():
    args = get_args()
    if args.smoke:
        args.epochs, args.bs, args.workers, args.warmup = 2, 64, 0, 1
        args.out = args.out if args.out != "runs/exp" else "runs/smoke"
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        amp_ctx = lambda: torch.autocast("cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
        print(f"device: {torch.cuda.get_device_name()} | AMP {amp_dtype}")
    else:
        import contextlib
        amp_ctx = contextlib.nullcontext
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        print(f"device: {device} (no AMP) - full training will be slow without a CUDA GPU")

    # ---- data
    x_all, y_all, x_test, test_ids = load_data(args.data, args.cache)
    tr_idx, va_idx = stratified_split(y_all, args.val_size, args.seed)
    if args.smoke:
        rng = np.random.RandomState(args.seed)  # split indices are grouped by class; sample across classes
        tr_idx = rng.permutation(tr_idx)[:512]
        va_idx = rng.permutation(va_idx)[:256] if len(va_idx) else va_idx
        x_test, test_ids = x_test[:128], test_ids[:128]
    x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
    x_va, y_va = x_all[va_idx], y_all[va_idx]
    print(f"train {len(y_tr)} | val {len(y_va)} | test {len(x_test)}")

    train_ds = AugDataset(x_tr, y_tr, augment=True, aug_policy=args.aug, gray_p=args.gray, jitter=args.jitter)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    prep = GPUPrep(device, channels_last=device.type == "cuda")

    # ---- model
    model = build_model(vars(args)).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"trainable parameters: {n_params:,}  (limit {PARAM_LIMIT:,})")
    if args.teacher_only:
        print("teacher_only: training a KD teacher; no submission will be written")
    else:
        assert n_params <= PARAM_LIMIT, "model exceeds the 5M parameter budget"
    teachers = [load_teacher(pth, device) for pth in args.teachers]
    if teachers:
        print(f"distilling from {len(teachers)} teacher(s): {args.teachers}  (alpha {args.kd_alpha}, T {args.kd_T})")
    fmodel = torch.compile(model) if args.compile else model
    ema = EMA(model, args.ema) if args.ema > 0 else None

    decay, no_decay = [], []
    for p_ in model.parameters():
        (no_decay if p_.ndim <= 1 else decay).append(p_)  # no weight decay on BN / bias
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=args.lr, momentum=args.momentum, nesterov=True)
    steps_per_epoch = len(loader)
    total, warm = args.epochs * steps_per_epoch, args.warmup * steps_per_epoch

    def lr_at(step):  # linear warm-up, then cosine to zero
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "n_params": n_params, "device": str(device)}, f, indent=2)

    # ---- train
    hist, best = [], {"acc": -1.0}
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = prep(xb), yb.to(device, non_blocking=True)
            if args.cutout > 0:
                xb = cutout(xb, args.cutout)
            mixed = args.cutmix > 0 and np.random.rand() < args.cutmix
            if mixed:
                xb, ya, yb2, lam = cutmix(xb, yb)
            with amp_ctx():
                out = fmodel(xb)
                loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb2) if mixed else crit(out, yb)
                if teachers:  # teachers see exactly the same augmented / mixed batch
                    T = args.kd_T
                    with torch.no_grad():
                        tp = torch.stack([F.softmax(t(xb).float() / T, 1) for t in teachers]).mean(0)
                    kd = F.kl_div(F.log_softmax(out.float() / T, 1), tp, reduction="batchmean") * T * T
                    loss = args.kd_alpha * kd + (1 - args.kd_alpha) * loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            loss_sum += loss.detach().float() * len(yb)
            correct += (out.argmax(1) == yb).sum()
            seen += len(yb)

        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": float(loss_sum) / seen,
               "train_acc": float(correct) / seen, "epoch_time_s": time.time() - t0}
        last = epoch == args.epochs
        if len(y_va) and (epoch % 5 == 0 or last or epoch <= 2):
            row["val_acc"] = float((predict_probs(model, x_va, prep, 1024, amp_ctx, False).argmax(1) == y_va).mean())
            if ema is not None:
                row["val_acc_ema"] = float((predict_probs(ema.module, x_va, prep, 1024, amp_ctx, False).argmax(1) == y_va).mean())
        hist.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        with open(os.path.join(args.out, "history.csv"), "w", newline="") as f:
            keys = ["epoch", "lr", "train_loss", "train_acc", "val_acc", "val_acc_ema", "epoch_time_s"]
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader(); w.writerows(hist)

    # ---- choose final weights (raw vs EMA) and evaluate
    candidates = {"raw": model}
    if ema is not None:
        candidates["ema"] = ema.module
    summary = {"n_params": n_params, "train_time_min": (time.time() - t_start) / 60}
    if len(y_va):
        for name, m in candidates.items():
            for tta in (False, True):
                acc = float((predict_probs(m, x_va, prep, 1024, amp_ctx, tta).argmax(1) == y_va).mean())
                summary[f"val_acc_{name}{'_tta' if tta else ''}"] = acc
                if not tta and acc > best["acc"]:
                    best = {"acc": acc, "name": name}
        final_name = best["name"]
    else:
        final_name = "ema" if ema is not None else "raw"
    final = candidates[final_name]
    summary["final_weights"] = final_name
    torch.save({"state_dict": final.state_dict(), "config": vars(args), "n_params": n_params},
               os.path.join(args.out, "model.pt"))

    if not args.teacher_only:  # an over-budget teacher must never produce a submission
        probs = predict_probs(final, x_test, prep, 1024, amp_ctx, tta=not args.no_tta)
        np.save(os.path.join(args.out, "test_probs.npy"), probs)
        pred = probs.argmax(1)
        with open(os.path.join(args.out, "submission.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            w.writerows(zip(test_ids.tolist(), pred.tolist()))
        summary["test_pred_class_counts"] = np.bincount(pred, minlength=10).tolist()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    plot_curves(hist, os.path.join(args.out, "curves.png"))
    print(f"done: outputs in {args.out}")


def plot_curves(hist, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [r["epoch"] for r in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [r["train_loss"] for r in hist])
    ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot(ep, [r["train_acc"] for r in hist], label="train (augmented)")
    for key, lab in (("val_acc", "val"), ("val_acc_ema", "val (EMA)")):
        pts = [(r["epoch"], r[key]) for r in hist if key in r]
        if pts:
            ax[1].plot(*zip(*pts), marker="o", ms=3, label=lab)
    ax[1].set(title="Accuracy", xlabel="epoch")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
