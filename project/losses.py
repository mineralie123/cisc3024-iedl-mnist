"""Clean implementation of I-EDL Eq. (5) and the matched MSE-EDL baseline.

I-MSE weights are trigamma(alpha_k), NOT diag(F): the rank-one contribution
cancels because both the one-hot target and Dirichlet sample sum to one.
The full Fisher matrix is used for the differentiable log determinant.
"""
import torch
from torch.nn import functional as F
from config import P, METHODS


def require_finite(name, value):
    if not torch.isfinite(value).all():
        detached = value.detach()
        finite = detached[torch.isfinite(detached)]
        limits = (float(finite.min()), float(finite.max())) if finite.numel() else None
        raise FloatingPointError(f"Nonfinite {name}: shape={tuple(value.shape)}, "
                                 f"finite_range={limits}, "
                                 f"bad_indices={(~torch.isfinite(detached)).nonzero()[:10].tolist()}")


def evidence_parameters(logits):
    z = logits.to(dtype=torch.float64)
    require_finite("logits", z)
    evidence = F.softplus(z)
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    q = alpha / S
    for name, tensor in (("evidence", evidence), ("alpha", alpha), ("S", S), ("q", q)):
        require_finite(name, tensor)
    return evidence, alpha, S, q


def prediction(logits, method):
    if method not in METHODS:
        raise ValueError(method)
    if method == "softmax":
        z = logits.to(torch.float64)
        require_finite("logits", z)
        logq = F.log_softmax(z, dim=-1)
        q = logq.exp()
        uncertainty = None
    else:
        _, alpha, S, q = evidence_parameters(logits)
        logq = alpha.log() - S.log()
        uncertainty = P.classes / S.squeeze(-1)
    require_finite("log probabilities", logq)
    require_finite("probabilities", q)
    return q, logq, uncertainty


def masked_kl(alpha, one_hot):
    a = alpha * (1.0 - one_hot) + one_hot
    total = a.sum(-1)
    K = alpha.shape[-1]
    kl = (torch.lgamma(total) - torch.lgamma(a).sum(-1)
          - torch.lgamma(a.new_tensor(float(K)))
          + ((a - 1.0) * (torch.digamma(a) - torch.digamma(total).unsqueeze(-1))).sum(-1))
    require_finite("masked KL", kl)
    return kl


def fisher_terms(alpha):
    trigamma = torch.polygamma(1, alpha)
    total_trigamma = torch.polygamma(1, alpha.sum(-1))
    fisher = torch.diag_embed(trigamma) - total_trigamma[:, None, None]
    require_finite("Fisher matrix", fisher)
    chol, info = torch.linalg.cholesky_ex(fisher, check_errors=False)
    if torch.any(info != 0):
        bad = (info != 0).nonzero().flatten().tolist()
        raise FloatingPointError(f"Fisher Cholesky failed: examples={bad}, "
                                 f"alpha_range=({alpha.detach().min().item()}, "
                                 f"{alpha.detach().max().item()}), info={info.tolist()}. "
                                 "No jitter or objective clamping applied.")
    negative_logdet = -2.0 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    require_finite("negative Fisher logdet", negative_logdet)
    return trigamma, fisher, negative_logdet


def objective(logits, labels, method, epoch):
    """Return batch mean loss and differentiable batch mean components.

epoch is zero-based: 0,1,...; no detached Fisher weights or epsilon shifts.
"""
    if method not in METHODS or epoch < 0:
        raise ValueError("Invalid method or epoch")
    if method == "softmax":
        loss = F.cross_entropy(logits.to(torch.float64), labels)
        require_finite("cross entropy", loss)
        return loss, {"cross_entropy": loss}
    _, alpha, S, q = evidence_parameters(logits)
    y = F.one_hot(labels, num_classes=P.classes).to(torch.float64)
    error = (y - q).square()
    variance = alpha * (S - alpha) / (S.square() * (S + 1.0))
    kl = masked_kl(alpha, y).mean()
    anneal = min(1.0, epoch / P.kl_anneal_epochs)
    components = {"masked_kl": kl, "annealed_kl": anneal * kl}
    if method == "edl":
        components.update(mse=error.sum(-1).mean(), variance=variance.sum(-1).mean())
        loss = components["mse"] + components["variance"] + components["annealed_kl"]
    else:
        weights, _, neg_logdet = fisher_terms(alpha)
        components.update(fisher_mse=(weights * error).sum(-1).mean(),
                          fisher_variance=(weights * variance).sum(-1).mean(),
                          negative_fisher_logdet=neg_logdet.mean(),
                          weighted_negative_fisher_logdet=P.fisher_coefficient * neg_logdet.mean())
        loss = (components["fisher_mse"] + components["fisher_variance"]
                + components["weighted_negative_fisher_logdet"] + components["annealed_kl"])
    for name, value in {"objective": loss, **components}.items():
        require_finite(name, value)
    return loss, components
