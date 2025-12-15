import os
import time

from src.faiss_client import FAISSClient
from src.llm_client import LLMClient
from src.pdf_processor import PDFProcessor

import logging

from utils.rag_utils import RAGUtils

logger = logging.getLogger(__name__)

class RAG:

    def __init__(self, config):
        self.config = config

        self.llm_client = LLMClient(self.config)
        self.pdf_processor = PDFProcessor(self.llm_client, self.config)
        self.faiss_client = FAISSClient(self.config)

        self.rag_utils = RAGUtils(self.config)

        self.chunk_batch_limit = self.config.chunk_batch_limit
        self.seconds_between_chunks_batches = self.config.seconds_between_chunks_batches

    def ingest_pdfs(self, paths):
        """
        Method to ingest PDFs into the FAISS store. Since RPM is an issue, we split the PDF chunks into batches, which
        are added to the FAISS store one by one, with a delay between each batch.

        :param paths: Path, or list of paths, to the PDF file(s)
        :return: None
        """

        if isinstance(paths, str):
            paths = [paths]

        logger.info("Ingesting {} PDFs using chunk batch size {} and {} seconds between each group..."
                    "".format(len(paths), self.chunk_batch_limit, self.seconds_between_chunks_batches))

        for path in paths:

            # Get PDF name
            pdf_name = os.path.splitext(os.path.basename(path))[0]

            logger.info("Ingesting PDF {} at location {}..."
                        "".format(pdf_name, path))

            # First, extract chunks from the PDF, and split into batches
            chunks = self.pdf_processor.pdf_to_chunks(path)
            chunk_batches = [chunks[i:i + self.chunk_batch_limit] for i in range(0, len(chunks), self.chunk_batch_limit)]

            # Then, add each group to the FAISS store
            for chunk_batch_number, chunk_batch in enumerate(chunk_batches):
                logger.info("\tChunk group {} of {}...".format(chunk_batch_number + 1, len(chunk_batches)))
                self.faiss_client.add_to_faiss_store(chunk_batch)
                time.sleep(self.seconds_between_chunks_batches)

            # Finally, save the FAISS store to disk
            self.faiss_client.save_faiss_store()

            # And update the store path in the config
            self.config.add_ingested_pdf(pdf_name)

            print("Ingested PDFs: {}".format(self.config.ingested_pdfs))

    def get_context_and_answer(self, query):
        """
        Method to retrieve relevant chunks from the FAISS store and answer the query using the retrieved context

        :param query: Query, as string
        :return: answer as string, chunks as list of retrieved chunks
        """

        relevant_chunks = self.faiss_client.retrieve(query, top_n=5)
        return self.llm_client.answer_with_context(query, relevant_chunks)

    def get_context_and_answer_rrf(self, query):
        """
        Method to answer the query using the retrieved context, using query expansion with RRF (Reciprocal Rank Fusion)

        :param query: Query, as string
        :return: response as string, expansive queries as list of strings, contexts as list of retrieved chunks
        """

        top_n = 5
        expansive_queries = self.llm_client.generate_questions(query)
        all_queries = [query] + expansive_queries
        all_query_chunks = [self.faiss_client.retrieve(query, top_n=top_n) for query in all_queries]
        fused_queries = self.rag_utils.rrf_fusion(all_query_chunks, k=60, top_n=15)

        response, contexts = self.llm_client.answer_with_context(query, fused_queries)

        return response, expansive_queries, contexts