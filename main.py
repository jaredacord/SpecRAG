import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.pdf_processer import PDFProcesser
from config import LoggingConfig
from src.faiss_client import FAISSClient
from src.llm_client import LLMClient

import logging
logger = logging.getLogger(__name__)



class main():

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
            "data/pdfs/NVMe-Base-2.0d.pdf",
            #"data/pdfs/NVMe-MI-1.2c.pdf"
                ]
        for pdf in pdfs:
            self.faiss_client.ingest_pdf(pdf)

    def usage(self):

        query = "Which bits within the Abort command specify the command identifier?"
        relevent_chunks = self.faiss_client.retrieve(query, top_n=5)
        response = self.llm_client.answer_with_context(query, relevent_chunks)
        print("Query: {}\n".format(query))
        print("Answer: {}\n".format(response["answer"]))
        for i, chunk in enumerate(response["chunks"]):
            print("=====Relevent chunk {}=====".format(i+1))
            print(chunk.page_content+" .\n")



if __name__ == '__main__':
    main()


