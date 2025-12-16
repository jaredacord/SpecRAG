import logging
import os

logger = logging.getLogger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class FAISSClient:
    def __init__(self, config):

        self.config = config

        # Define the google generative AI embeddings
        self.embedding_model = self.config.embedding_model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model)

        # Get relevant config values
        self.chunk_batch_limit = self.config.chunk_batch_limit
        self.seconds_between_chunks_batches = self.config.seconds_between_chunks_batches

        # Define the FAISS store path, and load
        self.storage_path = self.config.faiss_storage_path
        self.store = self.load_faiss_store()

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

    def retrieve(self, query, top_n=3, metadata_filter=None):
        """
        Method to retrieve the top N chunks from the FAISS store based on a query.

        :param query: Query, as string
        :param top_n: Number of chunks to retrieve
        :param metadata_filter: Filter to apply to the metadata when retrieving chunks
        :return: List of top N chunks
        """

        results = self.store.similarity_search(query, k=top_n, filter=metadata_filter)
        logger.info("Retrieved {} chunks".format(top_n))
        return results