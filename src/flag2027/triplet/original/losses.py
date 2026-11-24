"""Within-source, original-row-positive bidirectional ranking objectives.

All arms draw full uniform candidate permutations in both directions, using
an explicit CPU generator. No gender reweighting or hard-negative mining.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class NegativePermutations:
    image_to_audio: torch.Tensor
    audio_to_image: torch.Tensor
    candidate_counts: torch.Tensor
    full_candidate_counts: torch.Tensor


def sample_negative_permutations(identities, sources, *, generator):
    if generator is None or str(generator.device) != "cpu":
        raise ValueError("An explicit CPU negative generator is required")
    labels = identities.detach().cpu()
    source = sources.detach().cpu()
    if labels.ndim != 1 or source.shape != labels.shape or len(labels) < 2:
        raise ValueError("Aligned nontrivial identity/source vectors required")
    different = labels[:, None].ne(labels[None, :])
    legal = different & source[:, None].eq(source[None, :])
    counts = legal.sum(1)
    width = int(counts.max())
    directions = []
    for _ in range(2):
        indices = torch.full((len(labels), width), -1, dtype=torch.long)
        for i in range(len(labels)):
            candidates = legal[i].nonzero(as_tuple=False).flatten()
            order = torch.randperm(len(candidates), generator=generator)
            indices[i, :len(candidates)] = candidates[order]
        directions.append(indices)
    return NegativePermutations(*directions, counts, different.sum(1))


def normalized_l2_matrix(face_unit, audio_unit):
    return torch.sqrt((2.0 - 2.0 * (face_unit @ audio_unit.T)).clamp_min(0.0))


def _gather(distance, indices, q):
    selected = indices[:, :q].to(device=distance.device)
    mask = selected.ge(0)
    return distance.gather(1, selected.clamp_min(0)), mask


def contrastive_loss(face_unit, audio_unit, identities, sources, kind,
                     candidates: Optional[NegativePermutations] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    if kind not in {"T_ALL", "T2", "T4", "T8", "T12", "T16", "NCE_ALL", "NCE_2", "NCE_4"}:
        raise ValueError(kind)
    if face_unit.ndim != 2 or audio_unit.shape != face_unit.shape:
        raise ValueError("Modalities must have equal [B,D] shape")
    labels, source = identities.to(face_unit.device), sources.to(face_unit.device)
    if labels.shape != (len(face_unit),) or source.shape != labels.shape:
        raise ValueError("Identity/source vectors must align with embeddings")
    different = labels[:, None].ne(labels[None, :])
    legal = different & source[:, None].eq(source[None, :])
    counts, full_counts = legal.sum(1), different.sum(1)
    valid_anchor = counts.gt(0)
    if candidates is None:
        raise ValueError("Every arm consumes explicit full negative permutations")
    if not torch.equal(candidates.candidate_counts, counts.cpu()) or not torch.equal(candidates.full_candidate_counts, full_counts.cpu()):
        raise ValueError("Candidate counts do not match current identity/source batch")
    distance = normalized_l2_matrix(face_unit, audio_unit)
    positive = distance.diagonal()
    q = None if kind.endswith("ALL") else int(kind.replace("NCE_", "").replace("T", ""))
    if q is None:
        values_f, mask_f = distance, legal
        values_a, mask_a = distance.T, legal.T
    else:
        values_f, mask_f = _gather(distance, candidates.image_to_audio, q)
        values_a, mask_a = _gather(distance.T, candidates.audio_to_image, q)
    zero = distance.sum() * 0.0
    detail = {}
    if kind.startswith("T"):
        hinge_f = F.relu(positive[:, None] - values_f + .2)
        hinge_a = F.relu(positive[:, None] - values_a + .2)
        # Preserve historical all-different-ID anchor mass, renormalizing only
        # when a within-source anchor has no legal negative. Zero hinges count.
        weight = full_counts.to(distance.dtype) * valid_anchor
        weight = weight / weight.sum().clamp_min(1)
        means_f = hinge_f.masked_fill(~mask_f, 0.).sum(1) / mask_f.sum(1).clamp_min(1)
        means_a = hinge_a.masked_fill(~mask_a, 0.).sum(1) / mask_a.sum(1).clamp_min(1)
        loss_f, loss_a = (weight * means_f).sum(), (weight * means_a).sum()
        total = 2. * (loss_f + loss_a)
        detail['image_active_rate'] = float((hinge_f[mask_f] > 0).float().mean()) if mask_f.any() else 0.
        detail['audio_active_rate'] = float((hinge_a[mask_a] > 0).float().mean()) if mask_a.any() else 0.
    else:
        # Candidate zero is always the original-row positive; padding is -inf.
        logits_f = torch.cat((-positive[:, None], -values_f.masked_fill(~mask_f, float('inf'))), 1)
        logits_a = torch.cat((-positive[:, None], -values_a.masked_fill(~mask_a, float('inf'))), 1)
        if valid_anchor.any():
            target = torch.zeros(int(valid_anchor.sum()), dtype=torch.long, device=distance.device)
            loss_f = F.cross_entropy(logits_f[valid_anchor], target)
            loss_a = F.cross_entropy(logits_a[valid_anchor], target)
        else:
            loss_f = loss_a = zero
        total = (loss_f + loss_a) / 2.
    def mean_or_zero(value, mask):
        return float(value[mask].mean()) if mask.any() else 0.
    with torch.no_grad():
        detail.update(image_anchor_loss=float(loss_f), audio_anchor_loss=float(loss_a),
            positive_distance=float(positive.mean()), all_negative_distance=mean_or_zero(distance, different),
            legal_negative_distance=mean_or_zero(distance, legal),
            selected_image_negative_distance=mean_or_zero(values_f, mask_f),
            selected_audio_negative_distance=mean_or_zero(values_a, mask_a), batch_rows=len(labels),
            legal_negative_pairs=int(legal.sum()), full_negative_pairs=int(different.sum()),
            excluded_cross_source_negative_pairs=int((different & ~legal).sum()),
            selected_image_negative_pairs=int(mask_f.sum()), selected_audio_negative_pairs=int(mask_a.sum()),
            anchors_without_legal_negatives=int((~valid_anchor).sum()),
            anchors_with_fewer_than_q=int(((counts < q) & valid_anchor).sum()) if q is not None else 0,
            ignored_same_identity_offdiag=int((~different).sum()) - len(labels))
    return total, detail
