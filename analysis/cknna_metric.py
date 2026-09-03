"""CKNNA: mutual-k-nearest-neighbor Centered Kernel Alignment.

Representation-similarity metric from the "Platonic Representation
Hypothesis" (Huh, Cheung, Wang, Isola; ICML 2024). Like CKA, it measures how
aligned two sets of representations are on the *same* N probe inputs, but
restricts the alignment computation to pairs that are mutual top-k nearest
neighbors in both kernels -- this makes it far less sensitive to a handful of
outlier points dominating the (biased) HSIC estimate than vanilla linear CKA.

Adapted (topk masking + unbiased-HSIC core) from the reference implementation
at https://github.com/minyoungg/platonic-rep (metrics.py, `AlignmentMetrics.cknna`),
trimmed to just the pieces this repo needs.
"""
from __future__ import annotations

import torch


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012, eq. 5)."""
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)
    value = (
        torch.sum(K_tilde * L_tilde.T)
        + torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2))
        - 2 * torch.sum(K_tilde @ L_tilde) / (m - 2)
    )
    return value / (m * (m - 3))


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Biased HSIC (trace form) -- equivalent to the numerator of vanilla CKA."""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cka(feats_A: torch.Tensor, feats_B: torch.Tensor, unbiased: bool = False) -> float:
    """Vanilla linear CKA, for reference alongside CKNNA."""
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk, hsic_ll, hsic_kl = hsic_fn(K, K), hsic_fn(L, L), hsic_fn(K, L)
    return (hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)).item()


def cknna(
    feats_A: torch.Tensor,
    feats_B: torch.Tensor,
    topk: int = 10,
    unbiased: bool = True,
) -> float:
    """Mutual-kNN CKA between two [N, D] representation matrices on N matched inputs.

    Args:
        feats_A, feats_B: [N, D_a] / [N, D_b] representations for the SAME N
            probe inputs (D_a and D_b need not match -- CKNNA compares
            pairwise-similarity kernels, not raw feature spaces). Rows should
            be L2-normalized so the linear kernel reduces to cosine similarity.
        topk: neighborhood size for the mutual-kNN mask (paper default: 10).
            Must be >= 2 and < N.
        unbiased: use the unbiased HSIC estimator (excludes the diagonal from
            both the neighbor search and the alignment sum); matches the
            paper's default.

    Returns:
        A scalar in roughly [-1, 1] (like CKA, but restricted to the
        mutual-neighborhood mass); 1.0 = identical neighborhood structure.
    """
    n = feats_A.shape[0]
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    if topk >= n:
        raise ValueError(f"topk ({topk}) must be < number of probe samples ({n})")

    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def masked_sim(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        if unbiased:
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        _, topk_K = torch.topk(K_hat, topk, dim=1)
        _, topk_L = torch.topk(L_hat, topk, dim=1)
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L, 1)
        mask = mask_K * mask_L  # mutual-neighbor pairs only

        hsic_fn = hsic_unbiased if unbiased else hsic_biased
        return hsic_fn(mask * K, mask * L)

    sim_kl = masked_sim(K, L)
    sim_kk = masked_sim(K, K)
    sim_ll = masked_sim(L, L)
    return (sim_kl / (torch.sqrt(sim_kk * sim_ll) + 1e-6)).item()
