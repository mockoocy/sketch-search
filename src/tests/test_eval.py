import torch
from sktr.vector import EvaluationStore


def _upsert_gallery(eval_store: EvaluationStore) -> None:
    # Two categories: A around [1,0], B around [0,1]
    vecs = torch.tensor(
        [
            [1.00, 0.00],  # A1
            [0.90, 0.10],  # A2
            [0.60, 0.40],  # A3
            [0.05, 0.95],  # B1
            [0.15, 0.85],  # B2
            [0.25, 0.75],  # B3
        ],
        dtype=torch.float32,
    )
    paths = [
        "A1.jpg",
        "A2.jpg",
        "A3.jpg",
        "B1.jpg",
        "B2.jpg",
        "B3.jpg",
    ]
    cats = ["A", "A", "A", "B", "B", "B"]
    eval_store.upsert(vecs, paths, cats)


def test_calculate_map_score(eval_store: EvaluationStore) -> None:
    _upsert_gallery(eval_store)

    query_batch = torch.tensor(
        [
            [0.95, 0.05],  # close to A,
            [0.10, 0.90],  # close to B,
            [0.45, 0.55],  # somewhat in between
        ]
    )

    map_score = eval_store.mean_average_precision_at_k(
        top_k=3,
        query_embeddings=query_batch,
        query_categories=["A", "B", "A"],
    )
    # TODO: Calculate the mAP by hand and compare.

    # For the first query, we expect to retrieve three
    # embeddings for category A.
    query_1_ap = 1 / 3 * (1 / 1 + 2 / 2 + 3 / 3)  # == 1
    # Similarily for the second query
    query_2_ap = 1 / 3 * (1 / 1 + 2 / 2 + 3 / 3)  # == 1
    # The 3rd one is a bit more complicated
    query_3_ap = 1 / 3 * (1 / 1 + 0 / 2 + 0 / 3)  # == 1/3

    expected_map = 1 / 3 * (query_1_ap + query_2_ap + query_3_ap)  # == 7/9

    assert abs(map_score - expected_map) < 1e-5
