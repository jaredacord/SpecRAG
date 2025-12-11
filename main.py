import time
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

    def start(self):
        self.tests()

    def setup(self):

        pdfs = [
            "data/pdfs/NVMe-Base-2.0d.pdf",
            #"data/pdfs/NVMe-MI-1.2c.pdf"
                ]
        for pdf in pdfs:
            self.faiss_client.ingest_pdf(pdf)

    def usage(self):

        #query = "Where can I look for more information on the path relates status code?"
        query = "What information does a smart log contain? How is a smart log obtained?"

        relevent_chunks = self.faiss_client.retrieve(query, top_n=5)
        response = self.llm_client.answer_with_context(query, relevent_chunks)
        print("Query: {}\n".format(query))
        print("Answer: {}\n".format(response["answer"]))
        for i, chunk in enumerate(response["chunks"]):
            print(chunk)

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
        number_of_pages = 2

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


