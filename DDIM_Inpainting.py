"""
DDIM_Inpainting.py
==================
Main script for PID-guided DDIM inpainting on the 2D Swiss Roll.

Full pipeline:
  1. Swiss Roll data generation + train/test split
  2. NoisePredictor training (frozen at test time)
  3. Rectangular inpainting mask: x ∈ (-0.5, 0.5), y ∈ (-2.0, 2.0)
  4. PID-guided DDIM sampling loop (Augmented MDP)
  5. Per-step real-time metrics logging (5 convergence curves)
  6. Final evaluation metrics
  7. Visualisation: GIF + PNG plots
"""

import os
import json
import time
import imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_swiss_roll

from typing import Optional

from DDIM.ForwardProcess import ForwardDiffusion
from DDIM.NoisePredictor import NoisePredictor
from DDIM.PIDController import PIDController
from DDIM.AugmentedMDP import AugmentedMDP
from DDIM.BoundaryMetrics import (
    MASK_X_MIN, MASK_X_MAX, MASK_Y_MIN, MASK_Y_MAX,
    in_rect_mask, find_boundary_indices,
    compute_all_final_metrics,
)


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # --- Swiss Roll ---
    N_SAMPLES    = 3000
    NOISE_SCALE  = 0.1
    TRAIN_RATIO  = 0.8
    RANDOM_SEED  = 42

    # --- Diffusion ---
    TIMESTEPS    = 200
    BETA_START   = 1e-4
    BETA_END     = 0.02
    DDIM_STEPS   = 50
    ETA          = 0.0          # deterministic

    # --- Training ---
    BATCH_SIZE   = 40
    LR           = 1e-3
    EPOCHS       = 30000
    USE_EMA      = True
    EMA_BETA     = 0.995

    # --- PID Controller ---
    KP           = 0.05
    KI           = 0.1
    KD           = 0.005
    GAMMA        = 0.9           # I-term decay
    M_CLAMP      = 1.0           # I-term saturation
    MU_EMA       = 0.9           # EMA smoothing coefficient
    BETA_SIGMOID = 5.0           # SNR gate steepness
    THETA_SIGMOID = 1.0          # SNR gate threshold

    # --- Output ---
    PLOT_DIR     = "Plot"
    LOG_DIR      = "logs"
    SEED         = 42


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data preparation
# =============================================================================

def generate_swiss_roll_data(n_samples: int, noise: float, seed: int):
    """Generate Swiss Roll data and normalise to zero-mean unit-variance."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    data = data[:, [0, 2]]                        # use (x, z) axes
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    return torch.tensor(data, dtype=torch.float32)


def create_inpainting_mask(
    x: torch.Tensor,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
):
    """
    Create train/test mask for rectangular inpainting region.
    Returns (mask_indices, nonmask_indices).
    """
    inside = in_rect_mask(x)
    mask_indices = torch.where(inside)[0].tolist()
    nonmask_indices = torch.where(~inside)[0].tolist()
    return mask_indices, nonmask_indices


# =============================================================================
# Training
# =============================================================================

def train_model(
    dataset: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> NoisePredictor:
    """Train NoisePredictor on Swiss Roll data."""
    print(f"[Train] {cfg.EPOCHS} epochs, batch_size={cfg.BATCH_SIZE}, lr={cfg.LR}")

    forward_diffusion = ForwardDiffusion(
        timesteps=cfg.TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END,
    ).to(device)

    model = NoisePredictor(input_dim=2, time_dim=32).to(device)

    # EMA wrapper
    class EMA:
        def __init__(self, m, beta=0.995):
            self.beta = beta
            self.step = 0
            self.ema = {k: v.clone() for k, v in m.state_dict().items()}

        def update(self, m):
            self.step += 1
            with torch.no_grad():
                for k, v in m.named_parameters():
                    self.ema[k] = self.ema[k] * self.beta + v * (1 - self.beta)

        def apply(self, m):
            m.load_state_dict(self.ema)

    ema = EMA(model, beta=cfg.EMA_BETA) if cfg.USE_EMA else None

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.MSELoss()
    timesteps = cfg.TIMESTEPS

    for epoch in range(cfg.EPOCHS):
        idx = torch.randint(0, len(dataset), (cfg.BATCH_SIZE,))
        x_0 = dataset[idx].to(device)
        t = torch.randint(0, timesteps, (cfg.BATCH_SIZE,), device=device)
        noise = torch.randn_like(x_0)
        x_t = forward_diffusion.q_sample(x_0, t, noise=noise)

        eps_pred = model(x_t, t)
        loss = loss_fn(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema:
            ema.update(model)

        if epoch % 5000 == 0 or epoch == cfg.EPOCHS - 1:
            print(f"  Epoch {epoch}/{cfg.EPOCHS} | Loss={loss.item():.6f}")
            if ema:
                ema.apply(model)

    return model


# =============================================================================
# PID-Guided DDIM Inpainting Loop
# =============================================================================

def build_ddim_timestep_sequence(timesteps: int, ddim_steps: int):
    """
    Build strictly decreasing DDIM timestep list so that every
    transition has the same gap (= step_size), including the last
    one before reaching 0.

    E.g. T=200, steps=50, step_size=4, remainder=3:
      Old: [199, 195, ..., 3, 0]  ← last gap = 3 (BAD)
      New: [196, 192, ..., 4, 0]  ← last gap = 4 (GOOD)
    """
    step_size = timesteps // ddim_steps          # 4
    remainder = (timesteps - 1) % step_size     # 199 % 4 = 3

    seq = []
    cur = timesteps - 1 - remainder             # start = 199 - 3 = 196
    while cur > 0:
        seq.append(cur)
        cur -= step_size
    seq.append(0)
    return seq


def run_inpainting(
    model: nn.Module,
    x_known: torch.Tensor,
    x_GT_masked: torch.Tensor,
    mask_indices: list,
    cfg: Config,
    device: torch.device,
    forward_diffusion: Optional[ForwardDiffusion] = None,
):
    """
    PID-guided DDIM inpainting loop, powered by AugmentedMDP.

    Core idea: The noise predictor only sees the MASKED region.
    DDIM reverse sampling is performed ONLY on masked points, driven by boundary
    gradient signals. This prevents the model from "resetting" to a full Swiss Roll.

    Augmented MDP state: S_t = [x_t; I_t; D_t; e_t; bar_e_t] ∈ R^10
    Action:           a_t = SNR_lock ⊙ (Kp·e_t + Ki·I_t + Kd·D_t)
    Environment:       frozen DDIM model applied to masked region only
    Reward:           r_t = -loss_boundary = -MSE(边界x̂₀ ↔ 最近邻已知点)

    Returns:
        x_inpainted: Final inpainted points for masked region
        history:     Dict of per-step logged scalars
    """
    pid = PIDController(
        Kp=cfg.KP,
        Ki=cfg.KI,
        Kd=cfg.KD,
        gamma=cfg.GAMMA,
        M=cfg.M_CLAMP,
        mu_ema=cfg.MU_EMA,
        beta_sigmoid=cfg.BETA_SIGMOID,
        theta_sigmoid=cfg.THETA_SIGMOID,
    )

    N_mask = len(mask_indices)
    alphas = (1.0 - torch.linspace(cfg.BETA_START, cfg.BETA_END, cfg.TIMESTEPS)).to(device)
    alpha_bar = torch.cumprod(alphas, dim=0)

    # ── x_init: start from global Gaussian noise ───────────────────────────
    # This makes the inpainting trajectory visibly begin from the DDIM prior
    # instead of clustering around the known-data mean.
    ddim_timesteps = build_ddim_timestep_sequence(cfg.TIMESTEPS, cfg.DDIM_STEPS)
    t_start = ddim_timesteps[0]
    x_init = torch.randn(N_mask, 2, device=device)

    ddim_timesteps = build_ddim_timestep_sequence(cfg.TIMESTEPS, cfg.DDIM_STEPS)
    step_size = cfg.TIMESTEPS // cfg.DDIM_STEPS

    # Re-compute boundary indices from x_init (not from x_known)
    boundary_indices = find_boundary_indices(x_init)

    # ── Build AugmentedMDP ─────────────────────────────────────────────
    mdp = AugmentedMDP(
        pid_controller=pid,
        x_known=x_known,
        boundary_indices=boundary_indices,
        x_GT_masked=x_GT_masked,
    )
    x_cur = mdp.reset(x_init)

    # History for convergence curves
    history = {
        "t_list": [],
        "r_t_list": [],
        "e_t_norm_list": [],
        "u_t_norm_list": [],
        "I_t_norm_list": [],
        "mse_t_list": [],
        "snr_lock_list": [],
        "a_t_norm_list": [],
        "bar_e_norm_list": [],
        "snr_scaling_list": [],
        "x_cur_frames": [],
        "frame_t_list": [],
    }
    history["x_cur_frames"].append(x_cur.detach().cpu().clone())
    history["frame_t_list"].append(t_start)

    print(f"[Inpainting] {len(ddim_timesteps)} DDIM steps | {N_mask} masked points")

    for step_idx, t in enumerate(ddim_timesteps):
        prev_t = max(0, t - step_size)

        # One MDP step: masked-region DDIM + PID steering + reward
        x_cur, r_t, info = mdp.step(
            x_mask=x_cur,
            model=model,
            alpha_bar=alpha_bar,
            t=t,
            prev_t=prev_t,
        )

        # Record history
        state_norms = mdp.get_state_norms()
        history["t_list"].append(t)
        history["r_t_list"].append(info["r_t"])
        history["e_t_norm_list"].append(info["e_t_norm"])
        history["u_t_norm_list"].append(info["u_t_norm"])
        history["I_t_norm_list"].append(state_norms["I_norm"])
        history["mse_t_list"].append(info["mse_t"])
        history["snr_lock_list"].append(info["snr_lock"])
        history["a_t_norm_list"].append(info["a_t_norm"])
        history["bar_e_norm_list"].append(info["bar_e_norm"])
        history["snr_scaling_list"].append(info["blend_lambda"])
        history["x_cur_frames"].append(x_cur.detach().cpu().clone())
        history["frame_t_list"].append(t)

    return x_cur.detach(), history


# =============================================================================
# Visualisation
# =============================================================================

def plot_inpainting_comparison(
    x_full: torch.Tensor,
    x_known: torch.Tensor,
    x_inpaint: torch.Tensor,
    mask_indices: list,
    save_path: str,
):
    """Static comparison: GT / Masked / Inpainted."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Ground Truth
    ax = axes[0]
    ax.scatter(x_full[:, 0], x_full[:, 1], alpha=0.5, s=10, c="steelblue")
    ax.set_title("Ground Truth (Swiss Roll)", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Masked
    ax = axes[1]
    x_masked = x_full.clone()
    x_masked[mask_indices] = float("nan")
    known_pts = x_full[~np.isnan(x_masked[:, 0].numpy())]
    ax.scatter(known_pts[:, 0], known_pts[:, 1], alpha=0.5, s=10, c="steelblue")
    ax.set_title(f"Masked Input ({len(mask_indices)} pts removed)", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Inpainted
    ax = axes[2]
    x_inpainted_full = x_full.clone()
    x_inpainted_full[mask_indices] = x_inpaint.cpu()
    ax.scatter(x_inpainted_full[:, 0], x_inpainted_full[:, 1], alpha=0.5, s=10, c="darkorange")
    ax.scatter(x_known[:, 0], x_known[:, 1], alpha=0.3, s=8, c="steelblue", label="known")
    ax.set_title("PID-Guided DDIM Inpainting", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("z")
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_convergence_curves(history: dict, save_path: str):
    """Plot 5 real-time convergence curves + SNR gate."""
    t_list = history["t_list"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plots = [
        ("Instantaneous Reward / Penalty  r_t", t_list, history["r_t_list"], "r_t", "red"),
        ("Gradient Norm  ||e_t||",          t_list, history["e_t_norm_list"], "||e_t||", "blue"),
        ("PID Control Norm  ||u_t||",        t_list, history["u_t_norm_list"], "||u_t||", "green"),
        ("Integral State Norm  ||I_t||",     t_list, history["I_t_norm_list"], "||I_t||", "purple"),
        ("Instantaneous MSE  MSE_t",          t_list, history["mse_t_list"],    "MSE_t",  "orange"),
        ("SNR Gate  SNR_lock",               t_list, history["snr_lock_list"],  "SNR_lock", "navy"),
    ]

    for ax, (title, xs, ys, label, color) in zip(axes, plots):
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=3)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("DDIM timestep t")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.suptitle("PID-Guided DDIM Inpainting — Convergence Curves", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def plot_guidance_evolution(history: dict, save_path: str):
    """Plot ||a_t|| = SNR_lock · ||u_t|| over DDIM steps."""
    t_list = history["t_list"]
    a_norm = history["a_t_norm_list"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_list, a_norm, color="crimson", linewidth=2, marker="o", markersize=3)
    ax.set_title("Guidance Vector Magnitude  ||a_t|| = SNR_lock · ||u_t||", fontsize=12)
    ax.set_xlabel("DDIM timestep t")
    ax.set_ylabel("||a_t||")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


def create_inpainting_gif(
    x_known: torch.Tensor,
    history: dict,
    x_full: torch.Tensor,
    mask_indices: list,
    save_path: str,
):
    """Create GIF showing inpainting progress over DDIM steps."""
    frames = []
    step_interval = max(1, len(history["x_cur_frames"]) // 30)

    x_known_np = x_known.cpu().numpy()

    for i, x_cur_frame in enumerate(history["x_cur_frames"][::step_interval]):
        fig, ax = plt.subplots(figsize=(6, 6))

        # Known region
        ax.scatter(x_known_np[:, 0], x_known_np[:, 1],
                   alpha=0.3, s=8, c="steelblue", label="known")

        # Current inpainted region
        x_cur_np = x_cur_frame.cpu().numpy()
        ax.scatter(x_cur_np[:, 0], x_cur_np[:, 1],
                   alpha=0.8, s=12, c="darkorange", label="inpainting")

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        step_num = i * step_interval
        frame_t_list = history.get("frame_t_list", history["t_list"])
        t_val = frame_t_list[step_num] if step_num < len(frame_t_list) else 0
        ax.set_title(f"PID-Guided DDIM Inpainting  t={t_val}", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(img)
        plt.close()

    last = frames[-1]
    for _ in range(15):
        frames.append(last)

    imageio.mimsave(save_path, frames, fps=8, loop=0)
    print(f"[Saved] {save_path}")


def plot_final_metrics_summary(
    metrics: dict,
    save_path: str,
):
    """Bar chart of all final evaluation metrics."""
    names = list(metrics.keys())
    values = [float(v) for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Final Evaluation Metrics — PID-Guided DDIM Inpainting", fontsize=13)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ts_start = int(time.time() * 1000)
    cfg = Config()
    seed_everything(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg.PLOT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────────────────
    print("\n[1] Generating Swiss Roll data ...")
    x_full = generate_swiss_roll_data(cfg.N_SAMPLES, cfg.NOISE_SCALE, cfg.SEED)
    mask_indices, nonmask_indices = create_inpainting_mask(
        x_full, MASK_X_MIN, MASK_X_MAX, MASK_Y_MIN, MASK_Y_MAX
    )
    x_known = x_full[nonmask_indices].to(device)
    x_GT_masked = x_full[mask_indices].to(device)

    print(f"  Total points: {cfg.N_SAMPLES}")
    print(f"  Masked (to inpaint): {len(mask_indices)}")
    print(f"  Known (non-masked): {len(nonmask_indices)}")

    # Build forward_diffusion once (needed for x_init initialization)
    forward_diff = ForwardDiffusion(
        timesteps=cfg.TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END,
    ).to(device)

    # ── 2. Train ───────────────────────────────────────────────────────────
    print("\n[2] Training NoisePredictor ...")
    model = train_model(x_full, cfg, device)
    model.eval()
    model.to(device)

    # ── 3. Inpainting ──────────────────────────────────────────────────────
    print("\n[3] PID-Guided DDIM Inpainting ...")
    x_inpainted, history = run_inpainting(
        model=model,
        x_known=x_known,
        x_GT_masked=x_GT_masked,
        mask_indices=mask_indices,
        cfg=cfg,
        device=device,
        forward_diffusion=forward_diff,
    )

    # ── 4. Final Metrics ────────────────────────────────────────────────────
    print("\n[4] Computing final metrics ...")
    x_inpaint_np = x_inpainted.cpu()
    x_nonmask_np = x_known.cpu()

    boundary_indices = find_boundary_indices(x_inpainted)

    metrics = compute_all_final_metrics(
        x_inpaint=x_inpaint_np,
        x_GT=x_GT_masked,
        x_known=x_known,
        x_nonmask=x_nonmask_np,
        boundary_indices=boundary_indices,
    )

    print("\n  Final Metrics:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.6f}")

    # ── 5. Visualisation ────────────────────────────────────────────────────
    print("\n[5] Generating visualisations ...")

    plot_inpainting_comparison(
        x_full=x_full,
        x_known=x_known,
        x_inpaint=x_inpainted,
        mask_indices=mask_indices,
        save_path=os.path.join(cfg.PLOT_DIR, "inpainting_comparison.png"),
    )

    plot_convergence_curves(
        history=history,
        save_path=os.path.join(cfg.PLOT_DIR, "convergence_curves.png"),
    )

    plot_guidance_evolution(
        history=history,
        save_path=os.path.join(cfg.PLOT_DIR, "guidance_vector_evolution.png"),
    )

    create_inpainting_gif(
        x_known=x_known,
        history=history,
        x_full=x_full,
        mask_indices=mask_indices,
        save_path=os.path.join(cfg.PLOT_DIR, "ddim_inpainting_process.gif"),
    )

    plot_final_metrics_summary(
        metrics=metrics,
        save_path=os.path.join(cfg.PLOT_DIR, "evaluation_metrics_summary.png"),
    )

    # ── 6. Log files ────────────────────────────────────────────────────────
    print("\n[6] Writing log files ...")

    # PID params
    pid_params = PIDController(
        Kp=cfg.KP, Ki=cfg.KI, Kd=cfg.KD,
        gamma=cfg.GAMMA, M=cfg.M_CLAMP,
        mu_ema=cfg.MU_EMA,
        beta_sigmoid=cfg.BETA_SIGMOID,
        theta_sigmoid=cfg.THETA_SIGMOID,
    ).to_dict()
    with open(os.path.join(cfg.LOG_DIR, "pid_params.json"), "w") as f:
        json.dump(pid_params, f, indent=2)
    print(f"[Saved] {cfg.LOG_DIR}/pid_params.json")

    # Detailed step log
    step_log = {
        "t": history["t_list"],
        "r_t": history["r_t_list"],
        "e_t_norm": history["e_t_norm_list"],
        "u_t_norm": history["u_t_norm_list"],
        "I_t_norm": history["I_t_norm_list"],
        "mse_t": history["mse_t_list"],
        "snr_lock": history["snr_lock_list"],
        "a_t_norm": history["a_t_norm_list"],
        "bar_e_norm": history["bar_e_norm_list"],
        "snr_scaling": history["snr_scaling_list"],
        "final_metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(os.path.join(cfg.LOG_DIR, "debug-inpainting.json"), "w") as f:
        json.dump(step_log, f, indent=2)
    print(f"[Saved] {cfg.LOG_DIR}/debug-inpainting.json")

    print(f"\n[DONE] Total elapsed: {(time.time()*1000 - ts_start)/1000:.1f}s")


if __name__ == "__main__":
    main()
