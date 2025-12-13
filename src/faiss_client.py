import logging
import os
import time

from src.pdf_processer import PDFProcesser

logger = logging.getLogger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import FAISSConfig


class FAISSClient:
    def __init__(self):

        # Initialize a PDF processor
        self.pdf_processer = PDFProcesser()

        # Define the google generative AI embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model=FAISSConfig.embedding_model)

        # Get relevant config values
        self.chunk_batch_limit = FAISSConfig.chunk_batch_limit
        self.seconds_between_chunks_batches = FAISSConfig.seconds_between_chunks_batches

        # Define the FAISS store path, and load
        self.storage_path = FAISSConfig.storage_path
        self.store = self.load_faiss_store()

    def ingest_pdf(self, path):
        """
        Method to ingest a PDF into the FAISS store. Since RPM is an issue, we split the PDF chunks into batches, which
        are added to the FAISS store one by one, with a delay between each batch.

        :param path: Path to the PDF file
        :return: None
        """

        logger.info("Ingesting PDF {} with chunk batch size {} and {} seconds between each group..."
                    "".format(path, self.chunk_batch_limit, self.seconds_between_chunks_batches))

        # First, extract chunks from the PDF, and split into groups
        chunks = self.pdf_processer.pdf_to_chunks(path)
        chunk_batches = [chunks[i:i + self.chunk_batch_limit] for i in range(0, len(chunks), self.chunk_batch_limit)]

        # Then, add each group to the FAISS store
        for chunk_batch_number, chunk_batch in enumerate(chunk_batches):
            logger.info("\tChunk group {} of {}...".format(chunk_batch_number + 1, len(chunk_batches)))
            self.add_to_faiss_store(chunk_batch)
            time.sleep(self.seconds_between_chunks_batches)

        # Finally, save the FAISS store to disk
        self.save_faiss_store()

    def add_to_faiss_store(self, chunks):
        """
        Method to add chunks to the FAISS store.

        :param chunks: List of chunks to add
        :return: None
        """

        # Extract contents and metadatas from chunks
        contents = [chunk["page_content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        logger.info("\tAdding {} chunks to FAISS store...".format(len(chunks)))

        # Is store exists, add chunks to it, otherwise create a new store
        if self.store is not None:
            self.store.add_texts(texts=contents, embedding=self.embeddings, metadatas=metadatas)
            logger.info("\tFAISS store updated successfully.")

        else:
            self.store = FAISS.from_texts(texts=contents, embedding=self.embeddings, metadatas=metadatas)
            logger.info("\tFAISS store created and updated successfully.")

    def save_faiss_store(self):
        """
        Method to save the FAISS store to disk.

        :return: None
        """
        logger.info("Saving FAISS store to {}...".format(self.storage_path))
        self.store.save_local(self.storage_path)
        logger.info("FAISS store saved to {} successfully.".format(self.storage_path))

    def load_faiss_store(self):
        """
        Method to load the FAISS store from disk.
        :return: FAISS store object
        """

        logger.info("Loading FAISS store from {}...".format(self.storage_path))

        # If the FAISS store exists, load it, otherwise return None
        if os.path.exists(self.storage_path):
            store = FAISS.load_local(self.storage_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS store loaded successfully.")

        else:
            store = None
            logger.info("FAISS storage not found. Will create once data is added.")

        return store

    def retrieve(self, query, top_n=3):
        """
        Method to retrieve the top N chunks from the FAISS store based on a query.

        :param query: Query, as string
        :param top_n: Number of chunks to retrieve
        :return: List of top N chunks
        """

        results = self.store.similarity_search(query, k=top_n)
        logger.info("Retrieved {} chunks".format(top_n))
        return results