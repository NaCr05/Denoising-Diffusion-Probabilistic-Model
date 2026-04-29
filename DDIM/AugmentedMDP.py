"""
DDIM/AugmentedMDP.py
====================
Augmented MDP for global-noise DDIM inpainting.

The key design: at each DDIM step we operate on the FULL dataset (known + masked).
We let the noise predictor denoise the full dataset (it was trained on this), then
re-anchor the known pixels to their original clean values, and finally apply a
manifold pull only on the masked region.

Transition per step:
  x_full_ddim = DDIM_step(x_full_t, t→t-1)         ← full unconditional denoise
  x_full_t-1 = (known → original_clean) + (mask → manifold_pull(x_ddim_mask))

Reward:
  r_t = -loss_boundary = -MSE(边界inpainted ↔ 最近邻已知)
"""

import torch
from typing import Tuple, Optional, List

from DDIM.PIDController import PIDController
from DDIM.BoundaryMetrics import compute_boundary_loss


class AugmentedMDP:
    def __init__(
        self,
        pid_controller: PIDController,
        x_known: torch.Tensor,
        mask_indices: List[int],
        boundary_indices: Optional[List[int]] = None,
        x_GT_masked: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            pid_controller:  PID controller for history tracking
            x_known:         Known (non-masked) points, shape [N_known, 2]
            mask_indices:    Indices of masked points in the FULL dataset
            boundary_indices: Indices within mask region near the boundary
            x_GT_masked:     Ground truth for masked region (for mse_t logging)
        """
        self.pid = pid_controller
        self.x_known = x_known
        self.mask_indices = mask_indices              # e.g. [0, 5, 12, ...] in full data
        self.boundary_indices = boundary_indices or []  # within mask_indices
        self.x_GT_masked = x_GT_masked              # [N_mask, 2]

        self._x_cur_full: Optional[torch.Tensor] = None  # [N_full, 2]
        self._step_count: int = 0

    def reset(self, x_init_full: torch.Tensor) -> torch.Tensor:
        """Reset MDP with full initial state (global noise)."""
        self.pid.reset()
        self._x_cur_full = x_init_full.detach().clone()
        self._step_count = 0
        return self._x_cur_full

    def step(
        self,
        x_full: torch.Tensor,
        model: torch.nn.Module,
        alpha_bar: torch.Tensor,
        t: int,
        prev_t: int,
    ) -> Tuple[torch.Tensor, float, dict]:
        """
        One DDIM step on the full dataset with known-pixel anchoring.

        Pipeline:
          1. DDIM one-step on full noisy data → unconditional denoise
          2. Reset known pixels to their original clean values (anchor)
          3. Manifold pull on masked region only (boundary-guided)

        Args:
            x_full:    Current full state (known + masked), shape [N_full, 2]
            model:     Frozen DDIM noise predictor
            alpha_bar: Cumulative alpha bar, shape [T]
            t:         Current DDIM timestep
            prev_t:    Previous DDIM timestep

        Returns:
            x_next_full: Next full state [N_full, 2]
            r_t:        Scalar reward (negative boundary loss)
            info:       Dict of per-step metrics
        """
        device = x_full.device
        N_full = x_full.shape[0]

        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_1m_ab_t = torch.sqrt((1.0 - ab_t).clamp(min=1e-8))
        sqrt_ab_prev = torch.sqrt(ab_prev)
        sqrt_1m_ab_prev = torch.sqrt((1.0 - ab_prev).clamp(min=1e-8))

        # ── Step 1: Full unconditional DDIM denoise ────────────────────
        with torch.no_grad():
            t_tensor = torch.full((N_full,), t, device=device, dtype=torch.long)
            eps_pred = model(x_full, t_tensor)

        xhat_0 = (x_full - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t.clamp(min=1e-8)

        # DDIM reverse step from xhat_0
        pred_dir = (x_full - sqrt_ab_t * xhat_0) / sqrt_1m_ab_t
        x_ddim = sqrt_ab_prev * xhat_0 + sqrt_1m_ab_prev * pred_dir   # [N_full, 2]

        # ── Step 2: Anchor known pixels back to original clean values ────
        x_known_original = self.x_known.to(device=x_ddim.device, dtype=x_ddim.dtype)
        nonmask_mask = self._build_nonmask_mask(N_full, device)  # [N_full] bool
        x_ddim[nonmask_mask] = x_known_original

        # ── Step 3: Manifold pull on masked region only ──────────────────
        # Extract masked region after anchoring
        x_mask_ddim = x_ddim[self.mask_indices]    # [N_mask, 2]

        # λ adapts with noise level: high noise → larger pull
        # Using 1 - ab_t ≈ 0.02-0.15: small but meaningful pull
        # We scale it up: λ = clamp((1-ab_t) / 0.1, 0, 1) ∈ [0, 1]
        blend_lambda = float(torch.clamp((1.0 - ab_t) / 0.1, 0.0, 1.0).item())

        with torch.no_grad():
            nn_idx = torch.cdist(x_mask_ddim, self.x_known).argmin(dim=1)
            x_nearest = self.x_known[nn_idx]

        #x_mask_corrected = (1.0 - blend_lambda) * x_mask_ddim + blend_lambda * x_nearest
        

          # ── Step 4: Boundary loss on corrected masked region ────────────
        xhat_0_mask = xhat_0[self.mask_indices]   # [N_mask, 2]
        loss_boundary = self._compute_boundary_loss(xhat_0_mask)
        r_t = float(-loss_boundary.item()) if loss_boundary is not None else 0.0

        # ── Step 5: PID state update (history tracking) ──────────────────
        grad_per_point = self._get_boundary_grad_per_point(xhat_0_mask)
        e_t_scalar = grad_per_point.mean(dim=0, keepdim=True) if grad_per_point.numel() > 0 else torch.zeros(1, 2, device=device)
        ab_prev_for_pid = ab_prev if prev_t >= 0 else torch.tensor(1.0, device=device)
        a_t_dummy, u_t_dummy, snr_lock, bar_e_t, D_t = self.pid.compute_action(
            e_t_scalar, ab_t, ab_prev_for_pid
        )
    
        x_mask_corrected = x_mask_ddim + a_t_dummy
    
        # Rebuild full state, preserving the known-point anchors in x_ddim.
        x_next_full = x_ddim.clone()
        x_next_full[self.mask_indices] = x_mask_corrected 

      

        

        # ── Step 6: Build info dict ─────────────────────────────────────
        info = {
            "t": t,
            "prev_t": prev_t,
            "r_t": r_t,
            "e_t_norm": float(grad_per_point.norm().item()) if grad_per_point.numel() > 0 else 0.0,
            "u_t_norm": float(u_t_dummy.norm().item()),
            "a_t_norm": float(a_t_dummy.norm().item()),
            "snr_lock": float(snr_lock.item()),
            "bar_e_norm": float(bar_e_t.norm().item()),
            "D_t_norm": float(D_t.norm().item()),
            "blend_lambda": blend_lambda,
            "mse_t": float(
                ((x_mask_corrected - self.x_GT_masked) ** 2).mean().item()
                if self.x_GT_masked is not None
                else 0.0
            ),
        }

        self._x_cur_full = x_next_full.detach()
        self._step_count += 1
        return x_next_full, r_t, info

    def _build_nonmask_mask(self, N_full: int, device: torch.device) -> torch.Tensor:
        """Build a boolean mask [N_full] where True = known (non-masked)."""
        mask = torch.zeros(N_full, dtype=torch.bool, device=device)
        nonmask_indices = [i for i in range(N_full) if i not in self.mask_indices]
        mask[nonmask_indices] = True
        return mask

    def _compute_boundary_loss(self, xhat_0_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Boundary loss computed on the masked region's boundary vs known points."""
        if not self.boundary_indices or len(self.boundary_indices) == 0:
            return None
        # boundary_indices are offsets within mask_indices, not absolute indices
        return compute_boundary_loss(xhat_0_mask, self.x_known, self.boundary_indices)

    def _get_boundary_grad_per_point(
        self,
        xhat_0_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ∂L_boundary/∂xhat_0[i] for ALL masked points.
        Soft Gaussian nearest-neighbor with proximity-weighted broadcast.
        """
        if not self.boundary_indices or len(self.boundary_indices) == 0:
            return torch.zeros_like(xhat_0_mask)

        x_boundary = xhat_0_mask[self.boundary_indices]
        N_boundary = x_boundary.shape[0]

        sigma = 0.5
        dist_sq = torch.cdist(x_boundary, self.x_known) ** 2
        weights = torch.exp(-dist_sq / (2 * sigma ** 2))
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        x_nearest = weights @ self.x_known
        residual = x_boundary - x_nearest
        grad_boundary = 2.0 * residual / max(N_boundary, 1)

        dist_to_boundary = torch.cdist(xhat_0_mask, x_boundary) ** 2
        boundary_weights = torch.exp(-dist_to_boundary / (2 * sigma ** 2))
        boundary_weights = boundary_weights / (boundary_weights.sum(dim=1, keepdim=True) + 1e-8)
        grad_per_point = boundary_weights @ grad_boundary
        return grad_per_point

    def get_state_norms(self) -> dict:
        return self.pid.get_state_norms()
