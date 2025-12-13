import json
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.pdf_processer import PDFProcesser
from config import LoggingConfig, PDFProcessingConfig
from src.faiss_client import FAISSClient
from src.llm_client import LLMClient

import logging
logger = logging.getLogger(__name__)



class main:

    def __init__(self):
        self.llm_client = LLMClient()
        self.faiss_client = FAISSClient()

        self.setup_logging()
        self.start()

    def setup_logging(self):
        log_dir = Path(LoggingConfig.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"log_{time.strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=LoggingConfig.logging_level,
            format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This adds console output
            ]
        )

    def get_context_and_answer(self, query):
        """
        Method to retrieve relevant chunks from the FAISS store and answer the query using the retrieved context

        :param query: Query, as string
        :return: answer as string, chunks as list of retrieved chunks
        """

        relevant_chunks = self.faiss_client.retrieve(query, top_n=5)
        return self.llm_client.answer_with_context(query, relevant_chunks)

    def get_context_and_answer_rrf(self, query):

        top_n = 5
        expansive_queries = self.llm_client.generate_questions(query)
        all_queries = [query] + expansive_queries
        all_query_chunks = [self.faiss_client.retrieve(query, top_n=top_n) for query in all_queries]
        fused_queries = self.rrf_fusion(all_query_chunks, k=60, top_n=15)

        response, contexts = self.llm_client.answer_with_context(query, fused_queries)

        return response, expansive_queries, contexts

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

    def start(self):
        self.usage()

    def setup(self):

        pdfs = [
            #"data/pdfs/NVMe-Base-2.0d.pdf",
            #"data/pdfs/NVMe-Base-2.0d-small.pdf",
            #"data/pdfs/NVMe-MI-1.2c.pdf"
                ]
        for pdf in pdfs:
            self.faiss_client.ingest_pdf(pdf)

    def usage(self):

        #query = "Where can I look for more information on the path relates status code?"
        #query = "What values can I use for a fw commit action?"
        query = "How do I commit firmware that has been downloaded to an SSD?"

        print("========= Without Query Expansion =========")
        response, contexts = self.get_context_and_answer(query)
        print("Query: {}\n".format(query))
        print("Answer: {}\n".format(response))

        print("=====Relevant chunks====")
        for i, context in enumerate(contexts):
            print(json.dumps(context.metadata))


        print("\n\n========= With Query Expansion =========")
        response, expansive_queries, contexts = self.get_context_and_answer_rrf(query)
        print("Query: {}\n".format(query))
        for i, query in enumerate(expansive_queries, start=1):
            print("Additional Query Generated {}: {}".format(i, query))
        print("Answer: {}\n".format(response))

        print("=====Relevant chunks====")
        for i, context in enumerate(contexts):
            print(json.dumps(context.metadata))

    def tests(self):
        pdf_processer = PDFProcesser()

        #pdf = "data/pdfs/PCI_Express_5.0.pdf"
        #pdf = "data/pdfs/PCI_Express_5.0_image_test.pdf"
        pdf = "data/pdfs/NVMe-Base-2.0d-small.pdf"
        #pdf = "data/pdfs/NVMe-Base-2.0d-image-test.pdf"
        #pdf = "data/pdfs/SinglePage.pdf"

        chunks = pdf_processer.pdf_to_chunks(pdf)

        chunks_by_page = []
        start_page = 1
        number_of_pages = 15

        for i in range(start_page, start_page+number_of_pages):
            chunks_by_page.append(chunk for chunk in chunks if chunk["metadata"]["page"] == i)

        for i, chunk_by_page in enumerate(chunks_by_page, start=start_page):
            print(f"=== Page {i} =====")

            tables = []
            drawings = []
            texts = []

            for chunk in chunk_by_page:
                if chunk["metadata"]["type"] == "table":
                    tables.append(chunk)
                elif chunk["metadata"]["type"] == "drawing":
                    drawings.append(chunk)
                else:
                    texts.append(chunk)

            print(f"===== Page {i} - tables =====")
            for chunk in tables:
                print(chunk)

            print(f"===== Page {i} - drawings =====")
            for chunk in drawings:
                print(chunk)

            print(f"===== Page {i} - texts =====")
            for chunk in texts:
                print(chunk)

            print("\n")

    def test_generate_questions(self):
        query = "Which section contains information on the arbitration FID?"

        additional_queries = self.llm_client.generate_questions(query)
        for i, query in enumerate(additional_queries, start=1):
            print(f"Question {i}: {query}")

    def test_describe_image(self):
        image_path = "data/pdf_assets/NVMe-Base-2.0d/images/NVMe-Base-2.0d_p14_drawing0.png"

        description = self.llm_client.describe_image(image_path, PDFProcessingConfig.chunk_size)

        print(description)



if __name__ == '__main__':
    main()


