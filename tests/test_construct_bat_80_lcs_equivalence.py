"""
Equivalence test: refactored construct_bat_80_lcs vs. the original.

We import the new implementation from the pipeline and reimplement the
original here locally so we can run both against identical inputs and
assert bit-exact outputs across the edge cases that matter:

  1. Right-handed hitter, multi-frame
  2. Left-handed hitter, multi-frame
  3. Single-frame (no temporal derivative)
  4. Degenerate frame (BAT_80 stationary -> Z-candidate norm ~ 0)
  5. Mixed valid + degenerate frames

If any of these fail, the refactor is NOT a safe drop-in and must be reverted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add the pipeline to the import path
PIPELINE = Path("/sessions/brave-loving-wright/mnt/Missed Distance Pipeline")
sys.path.insert(0, str(PIPELINE))

from biomech_functions.functions import (  # noqa: E402
    construct_bat_80_lcs as construct_new,
    SWEET_SPOTS,
    _normalize_vectors,
    _backfill_axis_candidates,
    _fallback_axis_candidates,
    _is_left_handed_hitter,
)


def construct_original(top, knob, handedness=None):
    """Exact copy of the pre-refactor implementation for regression baseline."""
    top_xyz = np.asarray(top, dtype=float)
    knob_xyz = np.asarray(knob, dtype=float)
    if top_xyz.ndim != 2 or top_xyz.shape[1] != 3:
        raise ValueError("top must have shape (n_frames, 3).")
    if knob_xyz.shape != top_xyz.shape:
        raise ValueError("knob must have the same shape as top.")

    bat_vector = top_xyz - knob_xyz
    bat_80 = knob_xyz + SWEET_SPOTS["K80"] * bat_vector

    y_hat = _normalize_vectors(top_xyz - bat_80)
    bat_temp = np.zeros_like(bat_80, dtype=float)
    if len(bat_80) > 1:
        bat_temp[1:] = bat_80[:-1] - bat_80[1:]
        bat_temp[0] = bat_temp[1]

    z_candidates = -np.cross(y_hat, bat_temp)
    z_candidates = _backfill_axis_candidates(z_candidates)
    missing_z = np.linalg.norm(z_candidates, axis=1) <= 1e-8
    if missing_z.any():
        z_candidates[missing_z] = _fallback_axis_candidates(y_hat[missing_z])

    z_hat = _normalize_vectors(z_candidates)
    if _is_left_handed_hitter(handedness):
        z_hat *= -1.0
    x_hat = _normalize_vectors(np.cross(y_hat, z_hat))
    if not _is_left_handed_hitter(handedness):
        x_hat *= -1.0

    r_bat80 = np.stack([x_hat, y_hat, z_hat], axis=2)
    return bat_80, r_bat80


def compare(label, top, knob, handedness):
    bat_old, r_old = construct_original(top, knob, handedness)
    bat_new, r_new = construct_new(top, knob, handedness)
    np.testing.assert_array_equal(bat_old, bat_new, err_msg=f"{label}: bat_80 mismatch")
    np.testing.assert_array_equal(r_old, r_new, err_msg=f"{label}: r_bat80 mismatch")
    print(f"  PASS  {label}  (bat_80 shape={bat_new.shape}, r shape={r_new.shape})")


def main():
    rng = np.random.default_rng(seed=42)
    n = 200

    # Synthesize a plausible swing trajectory: bat sweeps an arc.
    t = np.linspace(0, 1, n)
    knob = np.column_stack([
        0.05 * np.cos(2 * np.pi * t),
        0.05 * np.sin(2 * np.pi * t),
        1.0 + 0.1 * t,
    ])
    # TOP sits ~33 inches (0.84 m) along a rotating shaft direction
    shaft_dir = np.column_stack([
        np.cos(2 * np.pi * t + 0.3),
        np.sin(2 * np.pi * t + 0.3),
        0.05 * np.ones(n),
    ])
    shaft_dir /= np.linalg.norm(shaft_dir, axis=1, keepdims=True)
    top = knob + 0.84 * shaft_dir

    print("Running equivalence tests...")

    compare("RHH multi-frame, handedness=None", top, knob, None)
    compare("RHH multi-frame, handedness='R'", top, knob, "R")
    compare("LHH multi-frame, handedness='L'", top, knob, "L")
    compare("LHH multi-frame, handedness='LEFT'", top, knob, "LEFT")

    # Single-frame (no temporal derivative available)
    compare("Single-frame RHH", top[:1], knob[:1], "R")
    compare("Single-frame LHH", top[:1], knob[:1], "L")

    # Inject a degenerate frame: bat momentarily stationary
    top_deg = top.copy()
    knob_deg = knob.copy()
    top_deg[50] = top_deg[49]
    knob_deg[50] = knob_deg[49]
    compare("Degenerate frame RHH", top_deg, knob_deg, "R")
    compare("Degenerate frame LHH", top_deg, knob_deg, "L")

    # Multiple consecutive degenerate frames
    top_deg2 = top.copy()
    knob_deg2 = knob.copy()
    for i in range(60, 65):
        top_deg2[i] = top_deg2[59]
        knob_deg2[i] = knob_deg2[59]
    compare("Run of degenerate frames RHH", top_deg2, knob_deg2, "R")

    # Tiny perturbations to check numerical equivalence under noise
    top_noisy = top + rng.normal(scale=1e-6, size=top.shape)
    knob_noisy = knob + rng.normal(scale=1e-6, size=knob.shape)
    compare("Noisy RHH", top_noisy, knob_noisy, "R")
    compare("Noisy LHH", top_noisy, knob_noisy, "L")

    # Verify the resulting frame is a right-handed orthonormal basis
    _, r = construct_new(top, knob, "R")
    x, y, z = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    np.testing.assert_allclose(np.einsum("ij,ij->i", x, y), 0, atol=1e-10)
    np.testing.assert_allclose(np.einsum("ij,ij->i", y, z), 0, atol=1e-10)
    np.testing.assert_allclose(np.einsum("ij,ij->i", x, z), 0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-10)
    print("  PASS  Orthonormality (x.y=y.z=x.z=0, |x|=|y|=|z|=1)")

    print("\nAll equivalence checks passed -- refactor is bit-exact.")


if __name__ == "__main__":
    main()
