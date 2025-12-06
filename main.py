import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pymupdf
import pandas as pd

from src.pdf_processer import PDFProcesser
from config import LoggingConfig
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

    def start(self):
        self.usage()

    def setup(self):

        pdfs = [
            "data/pdfs/NVMe-Base-2.0d-small.pdf",
            #"data/pdfs/NVMe-MI-1.2c.pdf"
                ]
        for pdf in pdfs:
            self.faiss_client.ingest_pdf(pdf)

    def usage(self):

        #query = "Where can I look for more information on the path relates status code?"
        query = "How are I/O Command Set Specifications and nvme specifications related?"

        relevent_chunks = self.faiss_client.retrieve(query, top_n=5)
        response = self.llm_client.answer_with_context(query, relevent_chunks)
        print("Query: {}\n".format(query))
        print("Answer: {}\n".format(response["answer"]))
        for i, chunk in enumerate(response["chunks"]):
            print(chunk)

    def tests(self):
        pdf_processer = PDFProcesser()

        #pdf = "data/pdfs/NVMe-Base-2.0d.pdf"
        pdf = "data/pdfs/NVMe-Base-2.0d-small.pdf"
        #pdf = "data/pdfs/SinglePage.pdf"

        chunks = pdf_processer.pdf_to_chunks(pdf)

        chunks_by_page = []
        for i in range(1, 16):
            chunks_by_page.append(chunk for chunk in chunks if chunk["metadata"]["page"] == i)

        for i, chunk_by_page in enumerate(chunks_by_page, start=1):
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



if __name__ == '__main__':
    main()


