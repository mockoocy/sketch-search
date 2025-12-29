import torch
import pytest

from sktr.metrics import compute_map_at_k, compute_map_at_k_chunked


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ks", [(1,), (3,), (5,), (3, 5)])
def test_chunked_and_unchunked_map_equivalence(seed, ks):
    torch.manual_seed(seed)

    device = torch.device("cpu")

    num_queries = 32
    num_gallery = 256
    dim = 16
    num_classes = 8

    query_emb = torch.randn(num_queries, dim, device=device)
    gallery_emb = torch.randn(num_gallery, dim, device=device)

    query_emb = torch.nn.functional.normalize(query_emb, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, dim=1)

    query_labels = torch.randint(0, num_classes, (num_queries,), device=device)
    gallery_labels = torch.randint(0, num_classes, (num_gallery,), device=device)

    sims = query_emb @ gallery_emb.t()

    out_full = compute_map_at_k(
        sims=sims,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        ks=ks,
    )

    out_chunked = compute_map_at_k_chunked(
        query_emb=query_emb,
        query_labels=query_labels,
        gallery_emb=gallery_emb,
        gallery_labels=gallery_labels,
        ks=ks,
        chunk_size=7,
    )

    for k in ks:
        map_full, p_full = out_full[k]
        map_chunked, p_chunked = out_chunked[k]

        assert map_full == pytest.approx(map_chunked, rel=1e-6, abs=1e-7)
        assert p_full == pytest.approx(p_chunked, rel=1e-6, abs=1e-7)
