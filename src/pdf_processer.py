import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

import logging
logger = logging.getLogger(__name__)

from config import PDFProcessingConfig

class PDFProcesser():
    def __init__(self):
        logger.info("Initializing PDF Processer")

        self.chunk_size = PDFProcessingConfig.chunk_size
        self.chunk_overlap = PDFProcessingConfig.chunk_overlap

    def pdf_to_chunks(self, path):

        logger.info("Processing PDF {}...".format(path))

        text = ""

        with pdfplumber.open(path) as pdf:

            logger.info("Extracting text from {} pages...".format(len(pdf.pages)))

            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)

        logger.info("Extracted {} chunks, with size {} and overlap {}."
                      "".format(len(chunks), self.chunk_size, self.chunk_overlap))

        return chunks
