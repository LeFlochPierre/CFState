"""Frozen iModulon projection mathematics shared across CF-State workflows.

Arrays are position-aligned by the caller: genes x components and genes x
samples for projection; components x samples for residualisation.
These functions do not estimate new iModulons or calibrate from test data.
"""
from __future__ import annotations

import numpy as np


def _matrix(value: object, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(value, dtype=float)
    was_vector = arr.ndim == 1
    if was_vector:
        arr = arr[:, None]
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError(name + " must be a finite 1D or 2D array")
    return arr, was_vector


def project_calibrated(loadings: object, expression_contrasts: object,
                       slopes: object) -> np.ndarray:
    """Return calibrated coordinates using pinv(M) @ delta_expression.

    Intercepts cancel ONLY for input expression contrasts. This is the
    exact algebra used in frozen audits 04k4, 07h and 08g.
    """
    m, m_vector = _matrix(loadings, "loadings")
    if m_vector:
        raise ValueError("loadings must be gene-by-component 2D")
    x, x_vector = _matrix(expression_contrasts, "expression_contrasts")
    s = np.asarray(slopes, dtype=float)
    if s.ndim != 1 or len(s) != m.shape[1] or not np.isfinite(s).all():
        raise ValueError("slopes must be finite and aligned to loading columns")
    if m.shape[0] != x.shape[0]:
        raise ValueError("expression gene order/size does not match loadings")
    # NumPy default rcond and operation order intentionally match the audits.
    result = (np.linalg.pinv(m) @ x) * s[:, None]
    return result[:, 0] if x_vector else result


def residualize_controls(activities: object, control_basis: object) -> np.ndarray:
    """Remove least-squares contributions of frozen controlled-state vectors.

    Each column of control_basis is a state vector. The caller is responsible
    for using the original frozen orientation and normalisation.
    """
    y, y_vector = _matrix(activities, "activities")
    b, b_vector = _matrix(control_basis, "control_basis")
    if b_vector:
        raise ValueError("control_basis must be a 2D array")
    if b.shape[0] != y.shape[0]:
        raise ValueError("control basis rows must match activity components")
    coefficients = np.linalg.lstsq(b, y, rcond=None)[0]
    result = y - b @ coefficients
    return result[:, 0] if y_vector else result


def counts_to_logtpm(counts: object, gene_lengths_bp: object) -> np.ndarray:
    """Use the exact length-normalised, per-panel TPM of audit 04k4.

    Requires counts of genes x samples and lengths in matching gene order.
    The denominator is recomputed on the supplied gene panel; do not mix
    full-gene and softcore TPM without explicitly preserving that choice.
    """
    c, c_vector = _matrix(counts, "counts")
    length = np.asarray(gene_lengths_bp, dtype=float)
    if length.ndim != 1 or length.shape[0] != c.shape[0]:
        raise ValueError("lengths must align to the gene rows")
    if not np.isfinite(length).all() or (length <= 0).any() or (c < 0).any():
        raise ValueError("counts must be nonnegative and lengths positive")
    rates = c / (length[:, None] / 1000.0)
    totals = rates.sum(axis=0)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("invalid TPM denominators")
    tpm = rates / totals[None, :] * 1e6
    if not np.allclose(tpm.sum(axis=0), 1e6, atol=0.01, rtol=0):
        raise ValueError("TPM columns do not sum to 1e6")
    logged = np.log2(tpm + 1.0)
    return logged[:, 0] if c_vector else logged


def calibrate_with_operator(operator: object, expression: object,
                            slopes: object) -> np.ndarray:
    """Use an already-frozen pseudoinverse; never refit it.

    The caller supplies gene-order-aligned expression and component-order
    slopes. Intercept cancellation is valid for *contrasts* only. This API
    also supports the historical raw, uncentred 08g perturbation projection.
    """
    op = np.asarray(operator, dtype=float)
    x = np.asarray(expression, dtype=float)
    s = np.asarray(slopes, dtype=float)
    if op.ndim != 2 or x.ndim not in (1, 2):
        raise ValueError("operator must be 2D and expression 1D or 2D")
    if x.shape[0] != op.shape[1]:
        raise ValueError("gene order/size does not match frozen operator")
    if s.ndim != 1 or len(s) != op.shape[0]:
        raise ValueError("slopes must align to operator rows")
    if not all(np.isfinite(value).all() for value in (op, x, s)):
        raise ValueError("operator, expression and slopes must be finite")
    raw = op @ x
    return raw * (s if x.ndim == 1 else s[:, None])


def project_residual(operator: object, expression_contrast: object,
                     slopes: object, stable_indices: object,
                     control_basis: object) -> np.ndarray:
    """Frozen iModulon projection -> stable subset -> controlled residual.

    Matches the two-stage formula in historic Phase 7/8 audits. Requires
    caller-provided fixed loading/operator, programme order, frozen slopes,
    stable indices, and already-normalised controlled-state basis.
    """
    activities = calibrate_with_operator(operator, expression_contrast, slopes)
    indices = np.asarray(stable_indices)
    if indices.ndim != 1 or indices.dtype.kind not in ("i", "u"):
        raise ValueError("stable_indices must be a 1D integer index array")
    if np.any(indices < 0) or np.any(indices >= activities.shape[0]):
        raise ValueError("stable_indices out of range")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("duplicate stable programme indices")
    stable = activities[indices]
    return residualize_controls(stable, control_basis)
