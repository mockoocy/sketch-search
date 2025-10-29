import numpy as np
from pymilvus import DataType, MilvusClient
from pathlib import Path

import torch

from sktr.type_defs import EmbeddingBatch, StoredImage

LOCAL_EMBEDDINGS_PATH = Path("./tmp/embeddings.db")
LOCAL_EMBEDDINGS_COLLECTION = "evaluation_collection"


class EvaluationStore:
    def __init__(
        self,
        embedding_size: int,
        store_path: Path = LOCAL_EMBEDDINGS_PATH,
    ) -> None:
        self.embedding_size = embedding_size
        self.store_path = store_path
        self._cat_counts: dict[str, int] = {}
        self.client = self._create_client()


    def __len__(self) -> int:
        stats = self.client.get_collection_stats(LOCAL_EMBEDDINGS_COLLECTION)
        return int(stats["row_count"])

    def _create_client(self) -> MilvusClient:
        local_embeddings_store = self.store_path

        if not local_embeddings_store.parent.exists():
            local_embeddings_store.parent.mkdir(parents=True, exist_ok=True)
            local_embeddings_store.touch()
        else:
            local_embeddings_store.touch(exist_ok=True)

        client = MilvusClient(local_embeddings_store.as_posix())

        if client.has_collection(LOCAL_EMBEDDINGS_COLLECTION):
            client.drop_collection(LOCAL_EMBEDDINGS_COLLECTION)

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            primary_field="path",
        )
        schema.add_field(
            "vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_size
        )
        schema.add_field("category", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field("path", datatype=DataType.VARCHAR, max_length=1024)

        client.create_collection(
            LOCAL_EMBEDDINGS_COLLECTION,
            dimension=self.embedding_size,
            metric_type="COSINE",
            schema=schema,
        )

        index_params = MilvusClient.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="FLAT",
        )

        client.create_index(
            collection_name=LOCAL_EMBEDDINGS_COLLECTION,
            index_params=index_params,
        )

        client.load_collection(LOCAL_EMBEDDINGS_COLLECTION)

        return client

    def _as_cpu_f32(self, embeddings: EmbeddingBatch) -> list[list[float]]:
        """Maps embeddings to a list of float32 values on CPU

        It is done, because Milvus prefers float32

        Args:
            embeddings: A PyTorch tensor of shape (batch_size, embedding_size)

        Returns:
            A list of lists containing the float32 values of the embeddings
        """
        assert embeddings.ndim == 2, "embeddings must be [B, D]"
        if embeddings.device.type != "cpu":
            embeddings = embeddings.cpu()
        return embeddings.detach().to(torch.float32).contiguous().tolist()

    def upsert(
        self,
        embeddings: EmbeddingBatch,
        paths: list[str],
        categories: list[str],
    ) -> None:
        """Inserts embedding to milvus store

        Args:
            embeddings: a PyTorch tensor of shape (batch_size, embedding_size)
            paths: a list of paths to photos of each embedded item
        """
        embeddings_normalized = torch.nn.functional.normalize(embeddings, dim=1)
        records = self._as_cpu_f32(embeddings_normalized)

        data = [
            {
                "path": path,
                "vector": record,
                "category": category,
            }
            for record, path, category in zip(records, paths, categories)
        ]

        self.client.upsert(collection_name=LOCAL_EMBEDDINGS_COLLECTION, data=data)
        for category in categories:
            self._cat_counts[category] = self._cat_counts.get(category, 0) + 1

    def search(
        self,
        query_embeddings: EmbeddingBatch,
        top_k: int,
    ) -> list[list[StoredImage]]:
        """Searches for the top_k most similar embeddings in the milvus store

        Args:
            query_embeddings: a PyTorch tensor of shape (batch_size, embedding_size)
            top_k: the number of top similar embeddings to retrieve

        Returns:
            A list of lists containing the IDs of the top_k most similar embeddings for each query embedding
        """
        embeddings_normalized = torch.nn.functional.normalize(query_embeddings, dim=1)
        query_list = self._as_cpu_f32(embeddings_normalized)
        return self.client.search(
            collection_name=LOCAL_EMBEDDINGS_COLLECTION,
            anns_field="vector",
            data=query_list,
            limit=top_k,
            output_fields=["category", "path"],
            consistency_level="Strong",
        )


    def mean_average_precision_at_k(
        self, top_k: int, query_embeddings: EmbeddingBatch, query_categories: list[str]
    ) -> float:
        """Computes the mAP@K metric

        mAP@K is calculated as mean of AP(Q)@K across all queries.
        AP is given as sum of Precision@k for rank k <= K divided by
        number of all relevant images (or K, depending which is smaller).
        The ranks come from vector database search; 1st image is the one with
        highest similarity to the query.
        Precision@K is the ratio of relevant images retrieved,
        to the total number of images retrieved (num_relevant / K).

        For example, mAP@5 for two query vectors could be calculated as follows:
        - For the first query vector, assume the top 5 retrieved images are [A, B, C, D, E]
            and the relevant images are [A, C, D, X].
            Then, AP(Q1)@5 = (AP@1 + AP@3 + AP@4) / 4 = (1/1 + 2/3 + 3/4) / 4
        - For the second query vector, assume the top 5 retrieved images are [F, G, H, I, J]
            and the relevant images are [F, H].
            Then, AP(Q2)@5 = (1/1 + 2/3) / 2
        - Finally, mAP@5 = (AP(Q1)@5 + AP(Q2)@5) / 2

        Args:
            query_embeddings: Embeddings used to retrieve relevant photos
            query_categories: Category of each image in the query_embeddings batch
            top_k: The number of top results to consider for the metric

        Returns:
            The value of the mAP@K metric
        """

        n_gallery = len(self)
        if n_gallery == 0:
            return 0.0
        result_count = min(top_k, n_gallery)

        search_results = self.search(query_embeddings, top_k=result_count)

        category_matrix = np.array(
            [
                [neighbour["category"] for neighbour in query_result]
                for query_result in search_results
            ],
            dtype=object,
        )  # [len(search_results), result_count]
        query_category_vector = np.array(query_categories, dtype=object).reshape(
            len(search_results), 1
        ) # [len(search_results), 1]
        relevant_matrix = category_matrix == query_category_vector # [len(search_results), result_count]
        ranks_matrix = np.arange(1, result_count + 1, dtype=np.float32) # [result_count]


        precision_matrix = (
            relevant_matrix.cumsum(axis=1).astype(np.float32) / ranks_matrix
        )

        category_counts_vector = np.array(
            [self._cat_counts.get(cat, 0) for cat in query_categories]
        )

        average_precision_vector = (
            np.sum(relevant_matrix * precision_matrix, axis=1) / category_counts_vector
        )
        return average_precision_vector.mean()
