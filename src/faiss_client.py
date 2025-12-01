import logging
import os
import time

from src.pdf_processer import PDFProcesser

logger = logging.getLogger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import FAISSConfig


class FAISSClient():
    def __init__(self):

        logger.info("Initializing FAISS Client")

        logger.info("Loading embedding model {}...".format(FAISSConfig.embedding_model))
        self.embeddings = GoogleGenerativeAIEmbeddings(model=FAISSConfig.embedding_model)

        self.pdf_processer = PDFProcesser()

        self.chunk_group_limit = FAISSConfig.chunk_group_limit
        self.seconds_between_chunks_groups = FAISSConfig.seconds_between_chunks_groups

        self.storage_path = FAISSConfig.storage_path
        self.store = self.load_faiss_store()

    def ingest_pdf(self, path):
        logger.info("Ingesting PDF {}...".format(path))
        chunks = self.pdf_processer.pdf_to_chunks(path)

        chunk_groupings = [chunks[i:i + self.chunk_group_limit] for i in range(0, len(chunks), self.chunk_group_limit)]

        for chunk_group_number, chunk_group in enumerate(chunk_groupings):
            logger.info("Chunk group {} of {}...".format(chunk_group_number + 1, len(chunk_groupings)))
            self.add_to_faiss_store(chunk_group)
            time.sleep(self.seconds_between_chunks_groups)

    def add_to_faiss_store(self, chunks):

        logger.info("Adding {} chunks to FAISS store...".format(len(chunks)))
        if self.store is not None:
            self.store.add_texts(chunks, embedding=self.embeddings)
            logger.info("FAISS store updated successfully.")

        else:
            self.store = FAISS.from_texts(chunks, embedding=self.embeddings)
            logger.info("FAISS store created and updated successfully.")

        self.store.save_local(self.storage_path)
        logger.info("FAISS store saved to {} successfully.".format(self.storage_path))

    def load_faiss_store(self):

        logger.info("Loading FAISS store from {}...".format(self.storage_path))
        if os.path.exists(self.storage_path):
            store = FAISS.load_local(self.storage_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS store loaded successfully.")

        else:
            store = None
            logger.info("FAISS storage not found. Will create once data is added.")

        return store

    def retrieve(self, query, top_n=3):
        results = self.store.similarity_search(query, k=top_n)
        logger.info("Retrieved {} chunks".format(top_n))
        return results