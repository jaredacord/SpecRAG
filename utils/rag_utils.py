from collections import defaultdict


class RAGUtils:

    def __init__(self, config):
        self.config = config

    def rrf_fusion(self, ranked_lists, k=60, top_n=None):
        """
        Apply Reciprocal Rank Fusion (RRF) to multiple ranked retrieval result lists.

        Args:
            ranked_lists (List[List[Document]]):
                List of ranked lists from similarity_search.
            k (int):
                RRF constant (larger = less aggressive rank decay).
                Typical values: 50–60.
            top_n (int | None):
                Optionally limit output to top N fused results.

        Returns:
            List[Document]: RRF-ranked, deduplicated list of chunks.
        """
        scores = defaultdict(float)
        chunk_by_id = {}

        for results in ranked_lists:
            for rank, chunk in enumerate(results):
                uid = chunk.metadata.get("uuid")
                if uid is None:
                    raise ValueError("Chunk missing metadata['uuid']")

                # Store canonical chunk instance
                chunk_by_id[uid] = chunk

                # RRF score contribution
                scores[uid] += 1.0 / (k + rank + 1)

        # Sort by fused RRF score (descending)
        ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = [chunk_by_id[uid] for uid, _ in ranked_ids]

        if top_n is not None:
            fused = fused[:top_n]

        return fused