import logging
import os
import time

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

from config import PDFProcessingConfig

class PDFProcesser:
    def __init__(self):
        logger.info("Initializing PDF Processer")

        self.llm_client = LLMClient()

        self.chunk_size = PDFProcessingConfig.chunk_size
        self.chunk_overlap = PDFProcessingConfig.chunk_overlap
        self.table_ext = PDFProcessingConfig.table_ext
        self.img_ext = PDFProcessingConfig.img_ext
        self.table_header_height = PDFProcessingConfig.table_header_height
        self.drawing_header_height = PDFProcessingConfig.drawing_header_height
        self.table_render_dpi = PDFProcessingConfig.table_render_dpi
        self.drawing_render_dpi = PDFProcessingConfig.drawing_render_dpi
        self.min_image_size = PDFProcessingConfig.min_image_size

        self.tables_dir = os.path.join(PDFProcessingConfig.pdf_assets_dir, "tables")
        self.images_dir = os.path.join(PDFProcessingConfig.pdf_assets_dir, "images")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def generate_image_caption(self, image_path):

        return self.llm_client.summarize_image(image_path)

    def get_page_drawings(self, pdf_path, pdf_name, page_num, page):

        min_img_x, min_img_y = self.min_image_size

        drawings = page.get_drawings() or []
        chunks_to_return = []
        drawing_boundaries = []

        drawing_num = 0

        for drawing in drawings:

            drawing_name = f"{pdf_name}_p{page_num}_drawing{drawing_num}.{self.img_ext}"
            drawing_output_path = os.path.join(self.images_dir, drawing_name)

            drawing_coordinates = drawing['rect']

            x0 = drawing_coordinates[0]
            y0 = drawing_coordinates[1]
            x1 = drawing_coordinates[2]
            y1 = drawing_coordinates[3]

            if x1 - x0 < min_img_x or y1 - y0 < min_img_y:
                continue

            drawing_boundaries.append((x0, y0, x1, y1))

            y0 = max(0, y0 - self.drawing_header_height)

            drawing_rect = pymupdf.Rect(x0, y0, x1, y1)
            drawing_image = page.get_pixmap(clip=drawing_rect, dpi=self.drawing_render_dpi)
            drawing_image.save(drawing_output_path)

            text = self.generate_image_caption(drawing_output_path)

            chunks_to_return.append(
                {
                    "page_content": text,
                    "metadata": {
                        "type": "drawing",
                        "source": pdf_path,
                        "page": page_num,
                        "table_num": drawing_num,
                        "table_path": drawing_output_path
                    }
                }
            )

            drawing_num += 1

        return chunks_to_return, drawing_boundaries

    def get_page_tables(self, pdf_path, pdf_name, page_num, page):

        tables = page.find_tables() or []
        chunks_to_return = []
        table_boundaries = []

        for table_num, table in enumerate(tables):

            table_name = f"{pdf_name}_p{page_num}_table{table_num}.{self.table_ext}"
            table_output_path = os.path.join(self.tables_dir, table_name)

            table_coordinates = table.bbox

            x0 = table_coordinates[0]
            y0 = table_coordinates[1]
            x1 = table_coordinates[2]
            y1 = table_coordinates[3]

            table_boundaries.append((x0, y0, x1, y1))

            table_rect = pymupdf.Rect(x0, y0, x1, y1)
            header_rect = pymupdf.Rect(x0, y0-self.table_header_height, x1, y0)
            table_with_header_rect = pymupdf.Rect(x0, max(0, y0-self.table_header_height), x1, y1)

            table_header_text = page.get_text("text", clip=header_rect) or ""
            table_text = page.get_text("text", clip=table_rect) or ""

            table_text_clean = " ".join(table_text.split())

            table_text_chunks = self.text_splitter.split_text(table_text_clean)

            for i, table_text_chunk in enumerate(table_text_chunks):

                text = table_header_text + " - " + table_text_chunk

                chunks_to_return.append(
                    {
                        "page_content": text,
                        "metadata":{
                            "type": "table",
                            "source": pdf_path,
                            "page": page_num,
                            "table_num": table_num,
                            "table_path": table_output_path,
                            "chunk_index": i
                        }
                    }
                )

            table_image = page.get_pixmap(clip=table_with_header_rect, dpi=self.table_render_dpi)
            table_image.save(table_output_path)

        return chunks_to_return, table_boundaries

    def get_page_text(self, pdf_path, page_num, page, table_boundaries):

        chunks_to_return = []

        table_boundaries = sorted(table_boundaries, key=lambda x: x[1])

        page_rect = page.rect

        x0 = page_rect[0]
        y0 = page_rect[1]
        x1 = page_rect[2]

        page_text = []

        text_area_start_y = y0

        for table_boundary in table_boundaries:

            text_area_end_y = table_boundary[1]
            text_area = pymupdf.Rect(x0, text_area_start_y, x1, text_area_end_y)
            text = page.get_text("text", clip=text_area) or ""
            text_area_start_y = table_boundary[3]
            if text.strip():
                page_text.append(" ".join(text.split()))

        if len(table_boundaries) == 0:
            text = page.get_text("text") or ""
            page_text = [" ".join(text.split())]

        total_text = "\n".join(page_text)

        total_text_chunks = self.text_splitter.split_text(total_text)

        for i, text in enumerate(total_text_chunks):

            chunks_to_return.append(
                {
                    "page_content": text,
                    "metadata": {
                        "type": "text",
                        "source": pdf_path,
                        "page": page_num,
                        "chunk_index": i
                    }
                }
            )

        return chunks_to_return

    def pdf_to_chunks(self, pdf_path):

        start = time.time()

        logger.info("Processing PDF {}...".format(pdf_path))

        all_chunks = []

        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        logger.info("Extracting text and tables...")

        with pymupdf.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):

                drawing_chunks, drawing_boundaries = self.get_page_drawings(pdf_path, pdf_name, page_num, page)
                all_chunks.extend(drawing_chunks)
                continue

                table_chunks, table_boundaries = self.get_page_tables(pdf_path, pdf_name, page_num, page)
                all_chunks.extend(table_chunks)

                non_text_boundaries = drawing_boundaries + table_boundaries

                text_chunks = self.get_page_text(pdf_path, page_num, page, non_text_boundaries)
                all_chunks.extend(text_chunks)


        duration = time.time() - start

        logger.info("Extracted {} chunks in {} seconds".format(len(all_chunks), round(duration, 3)))

        return all_chunks
