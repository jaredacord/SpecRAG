import json
import time
from pathlib import Path

from dotenv import load_dotenv

from config.config_client import ConfigClient
from rag.rag import RAG

load_dotenv()

from src.pdf_processor import PDFProcessor

import logging
logger = logging.getLogger(__name__)

class main:

    def __init__(self):
        self.config = ConfigClient()
        self.rag = RAG(self.config)

        self.setup_logging()
        self.start()

    def setup_logging(self):
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"log_{time.strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=self.config.logging_level,
            format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
            encoding='utf-8',
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
            "data/pdfs/NVMe-Base-2.0d-small.pdf",
            "data/pdfs/PCI_Express_5.0_image_test.pdf",
            "data/pdfs/SinglePage.pdf"
            #"data/pdfs/NVMe-MI-1.2c.pdf"
                ]
        pdfs = "data/pdfs/NVMe-Base-2.0d.pdf"
        self.rag.ingest_pdfs(pdfs)

    def usage(self):

        #query = "Where can I look for more information on the path relates status code?"
        #query = "What values can I use for a fw commit action?"
        #query = "How do I commit firmware that has been downloaded to an SSD?"
        #query = "This is an unrelated question."
        #query = "How many temperature sensors does the smart log track?"
        #query = "Is the temperature given in the smart log in degrees Celsius or Fahrenheit?"
        #query = "What could cause an LBA overlap error when downloading FW?"
        #query = "How do you make sure a set features command can change?"
        #query = "Can you list the steps involved in a firmware download?"
        query = "What are the options when commiting a firmware?"
        query = "What is VPD data?"

        '''
        print("========= Without Query Expansion =========")
        response, contexts = self.get_context_and_answer(query)
        print("Query: {}\n".format(query))
        print("Answer: {}\n".format(response))

        print("=====Relevant chunks====")
        for i, context in enumerate(contexts):
            print(json.dumps(context.metadata))


        print("\n\n========= With Query Expansion =========")
        '''
        response, expansive_queries, contexts = self.rag.get_context_and_answer_rrf(query)
        print("Query: {}\n".format(query))
        for i, query in enumerate(expansive_queries, start=1):
            print("Additional Query Generated {}: {}".format(i, query))
        print("\nAnswer: {}\n".format(response))

        print("=====Relevant chunks====")
        for i, context in enumerate(contexts):
            print(json.dumps(context.metadata))

    def tests(self):
        pdf_processer = PDFProcessor()

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

        description = self.llm_client.describe_image(image_path, self.config.chunk_size)

        print(description)



if __name__ == '__main__':
    main()


