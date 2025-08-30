# train_bc_transformer_3to1_11in_8out.py
import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

IN_DIM  = 11
OUT_DIM = 8
PAST_N  = 3

# ----------------------- Dataset -----------------------
class NPZNextStepDataset(Dataset):
    def __init__(self, npz_path: str):
        self.path  = npz_path
        with np.load(self.path, allow_pickle=False) as f:
            self.keys = sorted(f.files)
            self.lens = {k: f[k].shape[0] for k in self.keys}

        self.index = []
        for k in self.keys:
            T = self.lens[k]
            for t in range(3, T):          # <-- 최소 3스텝 과거 확보
                self.index.append((k, t))  # tgt = t

        with np.load(self.path, allow_pickle=False) as f:
            all_obs = np.concatenate([f[k] for k in self.keys], axis=0).astype(np.float32)
        self.mean_in  = torch.from_numpy(all_obs.mean(0))
        self.std_in   = torch.from_numpy(all_obs.std(0) + 1e-6)
        self.mean_out = self.mean_in[:OUT_DIM].clone()
        self.std_out  = self.std_in[:OUT_DIM].clone()

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        k, tgt = self.index[i]             # tgt = t
        with np.load(self.path, allow_pickle=False) as f:
            ep = f[k]                      # (T,11)

            frames = []
            for dt in range(PAST_N, 0, -1):
                idx = tgt - dt             # t-3, t-2, t-1
                frames.append(ep[idx])

            x = torch.from_numpy(np.stack(frames, axis=0)).float()  # (3,11)
            y = torch.from_numpy(ep[tgt, :OUT_DIM]).float()         # (8,) = s_t[:8]
        return x, y

# ----------------------- Model: Transformer -----------------------
class TransformerForecast(nn.Module):
    def __init__(self, d_model=128, nhead=8, nlayers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(IN_DIM, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, PAST_N, d_model))  # learned positional embedding (3,d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, OUT_DIM),
        )

        # xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):            # x: (B,3,11)
        h = self.proj(x) + self.pos  # (B,3,d_model)
        h = self.enc(h)              # (B,3,d_model)
        h_last = h[:, -1, :]         # (B,d_model)
        return self.head(h_last)     # (B,8)

# ----------------------- Train -----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = NPZNextStepDataset(args.data)

    N = len(ds)
    n_val = max(1, int(N * args.val_ratio))
    n_tr  = N - n_val
    g = torch.Generator().manual_seed(0)
    ds_tr, ds_va = torch.utils.data.random_split(ds, [n_tr, n_val], generator=g)

    mean_in, std_in = ds.mean_in, ds.std_in
    mean_out, std_out = ds.mean_out, ds.std_out

    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, 0)   # (B,3,11)
        y = torch.stack(ys, 0)   # (B,8)
        mi, si = mean_in.to(x.device), std_in.to(x.device)
        mo, so = mean_out.to(x.device), std_out.to(x.device)
        x = (x - mi) / si
        y = (y - mo) / so
        return x, y

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True, collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    model = TransformerForecast(d_model=args.d_model, nhead=args.nhead, nlayers=args.layers, d_ff=args.ff, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "norm_stats.npz"),
             mean_in=mean_in.numpy(), std_in=std_in.numpy(),
             mean_out=mean_out.numpy(), std_out=std_out.numpy())

    best = math.inf
    for ep in range(1, args.epochs+1):
        model.train(); tr = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * x.size(0)
        tr /= len(dl_tr.dataset)

        model.eval(); va = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                va += loss_fn(model(x), y).item() * x.size(0)
        va /= len(dl_va.dataset)
        sch.step()
        print(f"epoch {ep:03d} | train {tr:.5f} | val {va:.5f} | lr {sch.get_last_lr()[0]:.2e}")

        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "cfg": vars(args)},
                       os.path.join(args.out, "best.pt"))

    torch.save({"model": model.state_dict(), "cfg": vars(args)},
               os.path.join(args.out, "last.pt"))
    print(f"done. best val {best:.5f}")

# ----------------------- Inference -----------------------
@torch.no_grad()
def predict_next8(obs_hist_3, ckpt_path, cpu=False):
    """
    obs_hist_3: (3,11) 최근 3스텝(원스케일). 필요 시 호출측에서 복제 패딩해 전달.
    반환: (8,) 원스케일.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TransformerForecast(d_model=cfg["d_model"], nhead=cfg["nhead"], nlayers=cfg["layers"], d_ff=cfg["ff"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    stats = np.load(os.path.join(cfg["out"], "norm_stats.npz"))
    mi = torch.from_numpy(stats["mean_in"]).to(device)
    si = torch.from_numpy(stats["std_in"]).to(device)
    mo = torch.from_numpy(stats["mean_out"]).to(device)
    so = torch.from_numpy(stats["std_out"]).to(device)

    x = torch.as_tensor(obs_hist_3, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,11)
    x = (x - mi) / si
    y = model(x)[0]
    y = y * so + mo
    return y.detach().cpu().numpy()

# ----------------------- Args -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="dataset_all_afterpregrasp_t3.npz")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--out", type=str, default="try7/next8")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
