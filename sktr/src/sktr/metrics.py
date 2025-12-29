import torch


@torch.no_grad()
def compute_map_at_k_chunked(
    query_emb: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_emb: torch.Tensor,
    gallery_labels: torch.Tensor,
    *,
    ks: tuple[int, ...] = (10, 30),
    chunk_size: int = 256,
):
    gallery_emb = gallery_emb.t()  # [D, G]

    stats = {k: {"ap": 0.0, "p": 0.0} for k in ks}
    n = query_emb.size(0)

    for i in range(0, n, chunk_size):
        q = query_emb[i : i + chunk_size]
        ql = query_labels[i : i + chunk_size]

        sims = q @ gallery_emb  # [B, G]
        topk = torch.topk(sims, k=max(ks), dim=1).indices

        retrieved = gallery_labels[topk]  # [B, K]
        rel = retrieved == ql.unsqueeze(1)

        for k in ks:
            rel_k = rel[:, :k]

            prec = rel_k.float().mean(dim=1).sum().item()

            csum = torch.cumsum(rel_k.int(), dim=1)
            ranks = torch.arange(1, k + 1, device=q.device).unsqueeze(0)
            ap = (csum / ranks * rel_k).sum(dim=1)

            denom = torch.bincount(gallery_labels)[ql].clamp(min=1)
            denom = torch.minimum(denom, torch.tensor(k, device=q.device))
            ap = (ap / denom).sum().item()

            stats[k]["p"] += prec
            stats[k]["ap"] += ap

        del sims, topk, retrieved, rel

    return {k: (stats[k]["ap"] / n, stats[k]["p"] / n) for k in ks}


@torch.no_grad()
def compute_map_at_k(
    sims: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    ks: tuple[int, ...] = (10, 30),
) -> dict[int, tuple[float, float]]:
    """
    Returns {k: (mAP@k, Precision@k)} for cosine similarity matrix sims [Q,G].
    """
    k_max = max(ks)
    topk = torch.topk(sims, k=k_max, dim=1).indices  # [Q,k_max]

    rel = query_labels.unsqueeze(1) == gallery_labels.unsqueeze(0)  # [Q,G]
    rel_at_kmax = torch.gather(rel, 1, topk)  # [Q,k_max]

    out: dict[int, tuple[float, float]] = {}
    for k in ks:
        rel_at_k = rel_at_kmax[:, :k]  # [Q,k]

        # Precision@k
        prec_k = (rel_at_k.float().sum(dim=1) / float(k)).mean().item()

        # mAP@k
        csum = torch.cumsum(rel_at_k.int(), dim=1)
        ranks = torch.arange(1, k + 1, device=sims.device).unsqueeze(0)
        prec_at_r = csum.float() / ranks.float()
        ap_num = (prec_at_r * rel_at_k.float()).sum(dim=1)

        denom = rel.sum(dim=1).float()
        denom = torch.minimum(denom, torch.tensor(float(k), device=sims.device))
        denom = torch.clamp_min(denom, 1.0)

        map_k = (ap_num / denom).mean().item()
        out[k] = (map_k, prec_k)
    return out
