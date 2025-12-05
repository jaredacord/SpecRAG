import logging
import os
import time

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PDFProcessingConfig
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

class PDFProcesser:

    def __init__(self):

        # Initialize LLM client
        self.llm_client = LLMClient()

        # Get relevent config values
        self.chunk_overlap = PDFProcessingConfig.chunk_overlap
        self.chunk_size = PDFProcessingConfig.chunk_size
        self.drawing_render_dpi = PDFProcessingConfig.drawing_render_dpi
        self.img_ext = PDFProcessingConfig.img_ext
        self.min_image_size = PDFProcessingConfig.min_image_size
        self.object_header_height = PDFProcessingConfig.object_header_height
        self.table_render_dpi = PDFProcessingConfig.table_render_dpi

        # Define output directories
        self.images_dir = os.path.join(PDFProcessingConfig.pdf_assets_dir, "images")
        self.tables_dir = os.path.join(PDFProcessingConfig.pdf_assets_dir, "tables")

        # Make the directories if they don't exist
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        # Define the text splitter, used in chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def get_image_description(self, image_path):
        """
        This method uses the LLM client to describe the image at the given path, is about self.chunk_size characters.

        :param image_path: image path, as string
        :return: LLM generated description of the image, as string
        """

        return self.llm_client.summarize_image(image_path, self.chunk_size)

    def get_page_drawings(self, pdf_path, pdf_name, page_num, page):
        """
        Method to extract and save drawings from a PDF page.
        Each drawing is saved as an image, and the corresponding chunk data is given as:
            {
                "page_content": <llm textual description of the drawing>,
                "metadata": {
                    "type": "drawing",
                    "source": <path to PDF file as str>,
                    "page": <page number as int>,
                    "drawing_num": <drawing number by page as int>,
                    "drawing_path": <path to saved image file>
                }
            }

        :param pdf_path: Path to PDF (used in metadata)
        :param pdf_name: Name of PDF (used in file naming)
        :param page_num: Doc page number (used in metadata), starting from 1
        :param page: Page object from pymupdf
        :return: (List of chunks, list of drawing boundaries)
        """

        # Get the min image size x, y
        min_img_x, min_img_y = self.min_image_size

        # Get drawings from page
        drawings = page.get_drawings() or []

        # Initialize chunks to return list, drawing boundaries list, and drawing number
        chunks_to_return = []
        drawing_boundaries = []
        drawing_num = 0

        for drawing in drawings:

            # Define the output path for the drawing image
            drawing_name = "{}_p{}_drawing{}.{}".format(pdf_name, page_num, drawing_num, self.img_ext)
            drawing_output_path = os.path.join(self.images_dir, drawing_name)

            # Get the drawing x0, y0, x1, y1 coordinates
            drawing_coordinates = drawing['rect']
            x0, y0, x1, y1 = drawing_coordinates[0:4]

            # Check if the drawing is too small
            if x1 - x0 < min_img_x or y1 - y0 < min_img_y:
                continue

            # Since the pdf parser sometimes returns sub-drawings, skip if there exists a 'parent' drawing. Since the
            # 'parent' drawing contains the sub-drawing, no data is lost
            drawing_overlap_found = False
            for drawing_boundary in drawing_boundaries:
                x0_boundary, y0_boundary, x1_boundary, y1_boundary = drawing_boundary
                if x0 >= x0_boundary and x1 <= x1_boundary and y0 >= y0_boundary and y1 <= y1_boundary:
                    drawing_overlap_found = True
                    break
            if drawing_overlap_found:
                continue

            # Add the drawing boundary to the list
            drawing_boundaries.append((x0, y0, x1, y1))

            # Modify y0 to include the header, and get 'cropping' rectangle
            y0 = max(0, y0 - self.object_header_height)
            drawing_rect = pymupdf.Rect(x0, y0, x1, y1)

            # Save picture as image
            drawing_image = page.get_pixmap(clip=drawing_rect, dpi=self.drawing_render_dpi)
            drawing_image.save(drawing_output_path)

            # Get LLM generated description of the drawing
            text = self.get_image_description(drawing_output_path)

            # Define the chunk data and append to the list
            chunks_to_return.append(
                {
                    "page_content": text,
                    "metadata": {
                        "type": "drawing",
                        "source": pdf_path,
                        "page": page_num,
                        "drawing_num": drawing_num,
                        "drawing_path": drawing_output_path
                    }
                }
            )

            drawing_num += 1

        # Return chunks, and drawing boundaries
        return chunks_to_return, drawing_boundaries

    def get_page_tables(self, pdf_path, pdf_name, page_num, page):

        tables = page.find_tables() or []
        chunks_to_return = []
        table_boundaries = []

        table_num = 0

        for table in tables:

            table_name = f"{pdf_name}_p{page_num}_table{table_num}.{self.img_ext}"
            table_output_path = os.path.join(self.tables_dir, table_name)

            table_coordinates = table.bbox

            x0, y0, x1, y1 = table_coordinates[0:4]

            table_overlap_found = False
            for table_boundary in table_boundaries:
                x0_boundary, y0_boundary, x1_boundary, y1_boundary = table_boundary
                if x0 >= x0_boundary and x1 <= x1_boundary and y0 >= y0_boundary and y1 <= y1_boundary:
                    table_overlap_found = True
                    break

            if table_overlap_found:
                continue

            table_boundaries.append((x0, y0, x1, y1))

            table_rect = pymupdf.Rect(x0, y0, x1, y1)
            header_rect = pymupdf.Rect(x0, y0-self.object_header_height, x1, y0)
            table_with_header_rect = pymupdf.Rect(x0, max(0, y0-self.object_header_height), x1, y1)

            table_header_text = page.get_text("text", clip=header_rect) or ""
            table_text = page.get_text("text", clip=table_rect) or ""

            table_text_chunks = self.text_splitter.split_text(table_text)

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

            table_num += 1

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

                table_chunks, table_boundaries = self.get_page_tables(pdf_path, pdf_name, page_num, page)
                all_chunks.extend(table_chunks)

                non_text_boundaries = drawing_boundaries + table_boundaries

                text_chunks = self.get_page_text(pdf_path, page_num, page, non_text_boundaries)
                all_chunks.extend(text_chunks)


        duration = time.time() - start

        logger.info("Extracted {} chunks in {} seconds".format(len(all_chunks), round(duration, 3)))

        return all_chunks
