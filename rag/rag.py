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
        """
        Initialization method for the RAG class.

        :param config: ConfigClient instance
        """
        self.config = config

        # Initialize clients
        self.llm_client = LLMClient(self.config)
        self.pdf_processor = PDFProcessor(self.llm_client, self.config)
        self.faiss_client = FAISSClient(self.config)

        # Initialize RAG utils
        self.rag_utils = RAGUtils(self.config)

        # Define status update callback
        self.status_callback = None
        self.error_callback = None

        # Get relevant config values
        self.chunk_batch_limit = self.config.chunk_batch_limit
        self.seconds_between_chunks_batches = self.config.seconds_between_chunks_batches

    def set_status_callback(self, status_update):
        """
        Method to set the status update callback

        :param status_update: Method to call with status updates
        :return: None
        """

        self.status_callback = status_update

    def set_error_callback(self, error_dialog):
        """
        Method to set the error callback

        :param error_dialog: Method to call with error messages
        :return: None
        """

        self.error_callback = error_dialog

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
            chunks = self.pdf_processor.pdf_to_chunks(path, self.status_callback, self.error_callback)
            if chunks is None:
                continue

            chunk_batches = [chunks[i:i + self.chunk_batch_limit] for i in range(0, len(chunks), self.chunk_batch_limit)]
            num_of_chunks = len(chunks)

            if self.status_callback is not None:
                self.status_callback("Ingesting PDF {} - Adding {} context chunks to FAISS store..."
                                   "".format(pdf_name, num_of_chunks))

            # Then, add each group to the FAISS store
            for chunk_batch_number, chunk_batch in enumerate(chunk_batches):
                logger.info("\tChunk batch {} of {}...".format(chunk_batch_number + 1, len(chunk_batches)))
                self.faiss_client.add_to_faiss_store(chunk_batch)
                time.sleep(self.seconds_between_chunks_batches)

                if self.status_callback is not None:
                    self.status_callback("Ingesting PDF {} - {} of {} context chunks added to FAISS store..."
                                       "".format(pdf_name,
                                                 min(num_of_chunks, (chunk_batch_number + 1) * self.chunk_batch_limit),
                                                 num_of_chunks))

            # Finally, save the FAISS store to disk
            self.faiss_client.save_faiss_store()

            # And update the store path in the config
            self.config.add_ingested_pdf(pdf_name)

            if self.status_callback is not None:
                self.status_callback("Done")

    def get_context_and_answer(self, query, metadata_filter=None):
        """
        Method to retrieve relevant chunks from the FAISS store and answer the query using the retrieved context

        :param query: Query, as string
        :param metadata_filter: Filter to apply to the metadata when retrieving chunks
        :return: answer as string, chunks as list of retrieved chunks
        """

        relevant_chunks = self.faiss_client.retrieve(query, top_n=5, metadata_filter=metadata_filter)
        return self.llm_client.answer_with_context(query, relevant_chunks)

    def get_context_and_answer_rrf(self, query, metadata_filter=None):
        """
        Method to answer the query using the retrieved context, using query expansion with RRF (Reciprocal Rank Fusion)

        :param query: Query, as string
        :param metadata_filter: Filter to apply to the metadata when retrieving chunks
        :return: response as string, expansive queries as list of strings, contexts as list of retrieved chunks
        """

        top_n = 5

        if self.status_callback is not None:
            self.status_callback("Expanding Query...")
        expansive_queries = self.llm_client.generate_questions(query)
        all_queries = [query] + expansive_queries

        if self.status_callback is not None:
            self.status_callback("Retrieving context chunks for {} queries...".format(len(all_queries)))
        all_query_chunks = [self.faiss_client.retrieve(query, top_n=top_n, metadata_filter=metadata_filter)
                            for query in all_queries]

        if self.status_callback is not None:
            self.status_callback("Fusing {} retrieved context chunks...".format(len(all_query_chunks)))
        fused_queries = self.rag_utils.rrf_fusion(all_query_chunks, k=60, top_n=15)

        if self.status_callback is not None:
            self.status_callback("Answering query using {} context chunks...".format(len(fused_queries)))
        response, contexts = self.llm_client.answer_with_context(query, fused_queries)

        if self.status_callback is not None:
            self.status_callback("Done")

        return response, expansive_queries, contexts