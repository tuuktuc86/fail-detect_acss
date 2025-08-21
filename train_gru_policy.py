# train_ssl_3to1_11in_8out.py
import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

IN_DIM  = 11
OUT_DIM = 8
PAST_N  = 3

class NPZNextStepDataset(Dataset):
    """
    각 에피소드 ep: (T,11)
    입력 x_t = [obs_{t-3}, obs_{t-2}, obs_{t-1}]  (t<3이면 최근 프레임 복제 패딩)
    타깃 y_t = obs_t[:8]
    유효 t 범위: 0..T-1, 단 마지막 t=T-1은 다음 스텝이 없으므로 제외 → 0..T-2
    """
    def __init__(self, npz_path: str):
        self.data   = np.load(npz_path)
        self.keys   = sorted(self.data.files)  # 'episode000', ...
        self.index  = []  # (key, t) where target is obs[t]
        for k in self.keys:
            ep = self.data[k]
            assert ep.ndim == 2 and ep.shape[1] == IN_DIM, f"{k} shape {ep.shape}"
            T = ep.shape[0]
            if T >= 2:
                # t = 0..T-2  (target obs[t+1], 하지만 window는 t의 과거 3스텝 → 편의상 target_idx = t+1로 정의)
                for t in range(0, T-1):
                    self.index.append((k, t+1))  # target index
        # 정규화 통계
        all_obs = np.concatenate([self.data[k] for k in self.keys], axis=0).astype(np.float32)
        self.mean_in = torch.from_numpy(all_obs.mean(axis=0))                    # (11,)
        self.std_in  = torch.from_numpy(all_obs.std(axis=0) + 1e-6)              # (11,)
        self.mean_out = self.mean_in[:OUT_DIM].clone()                           # (8,)
        self.std_out  = self.std_in[:OUT_DIM].clone()                            # (8,)

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        k, tgt = self.index[i]            # tgt in [1..T-1]
        ep = self.data[k]                 # (T,11)
        # 이전 3스텝 구성: [tgt-3, tgt-2, tgt-1], 부족분은 첫 프레임 복제로 채움
        frames = []
        for dt in range(PAST_N, 0, -1):
            idx = tgt - dt
            if idx < 0: idx = 0
            frames.append(ep[idx])
        x = torch.from_numpy(np.stack(frames, axis=0)).float()   # (3,11)
        y = torch.from_numpy(ep[tgt, :OUT_DIM]).float()          # (8,)
        return x, y

class GRUForecast(nn.Module):
    def __init__(self, hidden=256, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size=IN_DIM, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, OUT_DIM),
        )
    def forward(self, x):                 # x: (B,3,11)
        h, _ = self.gru(x)                # (B,3,H)
        return self.head(h[:, -1, :])     # (B,8)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ds = NPZNextStepDataset(args.data)

    # split
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

    model = GRUForecast(hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
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

@torch.no_grad()
def predict_next8(obs_hist_3, ckpt_path, cpu=False):
    """
    obs_hist_3: (3,11) 최근 3스텝(원스케일). 필요 시 호출측에서 복제 패딩해 전달.
    반환: (8,) 원스케일.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GRUForecast(hidden=cfg["hidden"], layers=cfg["layers"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # 🔥 stats는 바로 torch.tensor로 변환
    stats = np.load(os.path.join(cfg["out"], "norm_stats.npz"))
    mi = torch.tensor(stats["mean_in"], dtype=torch.float32, device=device)
    si = torch.tensor(stats["std_in"], dtype=torch.float32, device=device)
    mo = torch.tensor(stats["mean_out"], dtype=torch.float32, device=device)
    so = torch.tensor(stats["std_out"], dtype=torch.float32, device=device)

    # 입력 obs_hist_3 (3,11)
    x = torch.as_tensor(obs_hist_3, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,11)
    x = (x - mi) / si
    y = model(x)[0]
    y = y * so + mo
    return y.cpu().numpy()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="/AILAB-summer-school-2025/dataset_all.npz")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--out", type=str, default="ssl_runs/next8")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
