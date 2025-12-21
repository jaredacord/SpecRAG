from collections import defaultdict


class RAGUtils:

    def __init__(self, config):
        """
        Method to initialize the RAGUtils utility class.

        :param config: ConfigClient instance
        """
        self.config = config

    def rrf_fusion(self, ranked_lists, k=60, top_n=None):
        """
        Method to apply Reciprocal Rank Fusion (RRF) to multiple ranked retrieval result lists.

        :param ranked_lists: List of ranked lists from similarity_search, as list of lists of chunks
        :param k: RRF constant (larger = less aggressive rank decay), typically 50-60
        :param top_n: Number of chunks to return, or None to return all chunks
        :return: List of RFF-ranked chunks
        """

        # Initialize RRF score dict and chunk dict
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