import logging
import os
import time
import uuid

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.pdf_processor_utils import PDFProcessorUtils

logger = logging.getLogger(__name__)

class PDFProcessor:

    def __init__(self, llm_client, config):

        self.config = config

        # Set the LLM client
        self.llm_client = llm_client

        # Initialize pdf processor utils
        self.pdf_processor_utils = PDFProcessorUtils(self.config)

        # Get relevant config values
        self.chunk_overlap = self.config.chunk_overlap
        self.chunk_size = self.config.chunk_size
        self.drawing_render_dpi = self.config.drawing_render_dpi
        self.img_ext = self.config.img_ext
        self.min_img_x, self.min_img_y = self.config.min_image_size
        self.object_header_height = self.config.object_header_height
        self.object_footer_height = self.config.object_footer_height
        self.page_header_range = self.config.page_header_range
        self.page_footer_range = self.config.page_footer_range
        self.table_render_dpi = self.config.table_render_dpi
        self.max_image_discontinuity = self.config.max_image_discontinuity

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

        return self.llm_client.describe_image(image_path, self.chunk_size)

    def get_page_tables(self, pdf_assets_path, pdf_path, pdf_name, page_num, page):
        """
        Method to extract and save tables from a PDF page.
        Each table may be divided into multiple chunks, depending on the size of the table. The table header, if found,
        is included with the content of each chunk so as to retain the context of the table. All chunks of the same
        table maintain the same metadata, with the exception of the incremented 'chunk_index'.
        Each drawing is saved as an image, and the corresponding chunk data is given as:
            {
                "page_content": <text extracted from the table>,
                "metadata":{
                    "uuid": <generated uuid4 string>,
                    "type": "table",
                    "source": <path to PDF file as str>,
                    "pdf_name": <pdf_name as str>,
                    "page": <page number as int>,
                    "table_num": <table number by page as int>,
                    "table_path": <path to saved image file>,
                    "chunk_index": <index of chunk within table as int>
            }

        :param pdf_assets_path: Path to directory where tables will be saved
        :param pdf_path: Path to PDF (used in metadata)
        :param pdf_name: Name of PDF (used in file naming)
        :param page_num: Doc page number (used in metadata), starting from 1
        :param page: Page object from pymupdf
        :return: (List of chunks, list of table boundaries)
        """

        # Get the x0, y0, x1 coordinates of the whole page
        page_rect = page.rect
        x0_page, y0_page, x1_page, y1_page = page_rect[0:4]

        # Extract tables from page
        tables = page.find_tables() or []

        # Initialize chunks to return list, table boundaries list, and table number
        chunks_to_return = []
        table_boundaries = []
        table_num = 0

        # Define tables path, and create it if it doesn't exist
        tables_path = os.path.join(pdf_assets_path, "tables")
        os.makedirs(tables_path, exist_ok=True)

        for table in tables:

            # Skip table if it is not valid
            if not self.pdf_processor_utils.is_table_valid(table, page):
                continue

            # Define the output path for the table image
            table_name = f"{pdf_name}_p{page_num}_table{table_num}.{self.img_ext}"
            table_output_path = os.path.join(tables_path, table_name)

            # Get the table's x0, y0, x1, y1 coordinates
            table_coordinates = table.bbox
            x0, y0, x1, y1 = table_coordinates[0:4]

            # Since the pdf parser sometimes returns sub-tables as seperate tables, skip if there exists a 'parent'
            # table. Since the 'parent' table contains the sub-table, no data is lost
            table_overlap_found = False
            for table_boundary in table_boundaries:
                x0_boundary, y0_boundary, x1_boundary, y1_boundary = table_boundary
                if x0 >= x0_boundary and x1 <= x1_boundary and y0 >= y0_boundary and y1 <= y1_boundary:
                    table_overlap_found = True
                    break
            if table_overlap_found:
                continue

            # Extend table boundaries to fill page horizontally
            x0 = x0_page
            x1 = x1_page

            # Add the drawing boundary to the list
            table_boundaries.append((x0, y0, x1, y1))

            # Define the boundaries for the table, header, and table + header + footer
            table_rect = pymupdf.Rect(x0, y0, x1, y1)
            header_rect = pymupdf.Rect(x0, y0-self.object_header_height, x1, y0)
            table_with_header_footer_rect = pymupdf.Rect(x0, max(y0_page, y0 - self.object_header_height), x1, min(y1_page, y1 + self.object_footer_height))

            # Get the header text and table text
            table_header_text = page.get_text("text", clip=header_rect) or ""
            table_text = page.get_text("text", clip=table_rect) or ""

            # Split the table text into chunks
            table_text_chunks = self.text_splitter.split_text(table_text)

            for i, table_text_chunk in enumerate(table_text_chunks):

                # Preface each chunk's content with the table header
                text = table_header_text + " - " + table_text_chunk

                # Define the chunk data and append to the list
                chunks_to_return.append(
                    {
                        "page_content": text,
                        "metadata":{
                            "uuid": str(uuid.uuid4()),
                            "type": "table",
                            "source": pdf_path,
                            "pdf_name": pdf_name,
                            "page": page_num,
                            "table_num": table_num,
                            "table_path": table_output_path,
                            "chunk_index": i
                        }
                    }
                )

            # Save table as image (including header)
            table_image = page.get_pixmap(clip=table_with_header_footer_rect, dpi=self.table_render_dpi)
            table_image.save(table_output_path)

            table_num += 1

        # Just in case, remove empty chunks
        chunks_to_return = [chunk for chunk in chunks_to_return if chunk.get("page_content").strip()]

        # Return chunks, and table boundaries
        return chunks_to_return, table_boundaries

    def get_page_drawings(self, pdf_assets_path, pdf_path, pdf_name, page_num, page, table_boundaries):
        """
        Method to extract and save drawings from a PDF page.
        Each drawing is saved as an image, and the corresponding chunk data is given as:
            {
                "page_content": <llm textual description of the drawing>,
                "metadata": {
                    "uuid": <generated uuid4 string>,
                    "type": "drawing",
                    "source": <path to PDF file as str>,
                    "pdf_name": <pdf_name as str>,
                    "page": <page number as int>,
                    "drawing_num": <drawing number by page as int>,
                    "drawing_path": <path to saved image file>
                }
            }

        :param pdf_assets_path: Path to directory where drawings will be saved
        :param pdf_path: Path to PDF (used in metadata)
        :param pdf_name: Name of PDF (used in file naming)
        :param page_num: Doc page number (used in metadata), starting from 1
        :param page: Page object from pymupdf
        :param table_boundaries:
        :return: (List of chunks, list of drawing boundaries)
        """

        # Get the x0, y0, x1 coordinates of the whole page
        page_rect = page.rect
        x0_page, y0_page, x1_page, y1_page = page_rect[0:4]

        # Get drawings from page
        raw_drawings = page.get_drawings() or []

        # Initialize chunks to return list, drawing boundaries list, and drawing number
        chunks_to_return = []
        drawing_boundaries = []
        drawing_num = 0

        # Define drawings path, and create it if it doesn't exist
        drawings_path = os.path.join(pdf_assets_path, "images")
        os.makedirs(drawings_path, exist_ok=True)

        for drawing in raw_drawings:

            if not self.pdf_processor_utils.is_drawing_valid(drawing, page):
                continue

            drawing_coordinates = drawing['rect']
            x0, y0, x1, y1 = drawing_coordinates[0:4]

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

            drawing_boundaries.append([x0, y0, x1, y1])

        drawing_boundaries = self.pdf_processor_utils.remove_itersection(drawing_boundaries, table_boundaries)

        drawing_boundaries = self.pdf_processor_utils.merge_boxes_iterative(drawing_boundaries, self.max_image_discontinuity)

        for drawing in drawing_boundaries:

            # Define the output path for the drawing image
            drawing_name = "{}_p{}_drawing{}.{}".format(pdf_name, page_num, drawing_num, self.img_ext)
            drawing_output_path = os.path.join(drawings_path, drawing_name)

            x0, y0, x1, y1 = drawing

            # Extend table boundaries to fill page horizontally
            x0 = x0_page
            x1 = x1_page

            # Modify y0 to include the header, and get boundaries
            y0 = max(y0_page, y0 - self.object_header_height)
            y1 = min(y1_page, y1 + self.object_footer_height)
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
                        "uuid": str(uuid.uuid4()),
                        "type": "drawing",
                        "source": pdf_path,
                        "pdf_name": pdf_name,
                        "page": page_num,
                        "drawing_num": drawing_num,
                        "drawing_path": drawing_output_path
                    }
                }
            )

            drawing_num += 1

        # Just in case, remove empty chunks
        chunks_to_return = [chunk for chunk in chunks_to_return if chunk.get("page_content").strip()]

        # Return chunks, and drawing boundaries
        return chunks_to_return, drawing_boundaries

    def get_page_text(self, pdf_path, pdf_name, page_num, page, non_text_boundaries):
        """
        Method to extract text from a PDF page. Since the page content includes drawings, tables, and so on with
        textual aspects, we first filter out the non-textual portions of the page (since the information is already
        encoded). We then extract the remaining pure textual content, and split it into chunks.
        One assumption we make is that the non-textual portions of the page fill the page horizontally.
        Each text chunk is given as:
            {
                "page_content": <extracted text>,
                "metadata": {
                    "uuid": <generated uuid4 string>,
                    "type": "text",
                    "source": <path to PDF file as str>,
                    "pdf_name": <pdf_name as str>,
                    "page": <page number as int>,
                    "chunk_index": <index of chunk within page as int>
                }
            }

        :param pdf_path: Path to PDF (used in metadata)
        :param pdf_name: Name of PDF (used in metadata)
        :param page_num: Doc page number (used in metadata), starting from 1
        :param page: Page object from pymupdf
        :param non_text_boundaries: List of boundaries for non-textual portions of the page
        :return:
        """

        # Define the chunks to return list, and the list to store the text of each text area
        chunks_to_return = []
        page_text = []

        # Sort non-textual boundaries by y0
        non_text_boundaries = sorted(non_text_boundaries, key=lambda x: x[1])

        # Get the x0, y0, x1 coordinates of the whole page
        page_rect = page.rect
        x0, y0, x1, y1 = page_rect[0:4]

        # Restrict view to within page header and footer
        y0 += self.page_header_range
        y1 -= self.page_footer_range

        # Set the starting y for the text area at the top of the page (y0)
        text_area_start_y = y0

        for non_text_boundary in non_text_boundaries:

            # Get the starting y of the next non-text area, which is the ending y of the current text area
            text_area_end_y = non_text_boundary[1]

            # Define the text area, and extract the text
            text_area = pymupdf.Rect(x0, text_area_start_y, x1, text_area_end_y)
            text = page.get_text("text", clip=text_area) or ""

            # Update the starting y for the next text area
            text_area_start_y = non_text_boundary[3]

            # Only add the text if it is not empty (E.g., the empty space between two tables would produce this)
            if text.strip():
                page_text.append(text)

        # Get the last text block, or whole page text if there were no non-textual boundaries
        text_area = pymupdf.Rect(x0, text_area_start_y, x1, y1)
        text = page.get_text("text", clip=text_area) or ""
        if text.strip():
            page_text.append(text)

        # Combine text, and chunk
        total_text = "\n".join(page_text)
        total_text_chunks = self.text_splitter.split_text(total_text)

        # For each text chunk, define the chunk data and append to the list
        for i, text in enumerate(total_text_chunks):
            chunks_to_return.append(
                {
                    "page_content": text,
                    "metadata": {
                        "uuid": str(uuid.uuid4()),
                        "type": "text",
                        "source": pdf_path,
                        "pdf_name": pdf_name,
                        "page": page_num,
                        "chunk_index": i
                    }
                }
            )

        # Just in case, remove empty chunks
        chunks_to_return = [chunk for chunk in chunks_to_return if chunk.get("page_content").strip()]

        # Return chunks
        return chunks_to_return

    def pdf_to_chunks(self, pdf_path, status_update=None):
        """
        Method to extract content from a pdf file, save the assets, and return a list of chunks ready for the
        vector storage

        :param pdf_path: Path to PDF
        :return: List of chunks
        """

        # Start the timer
        start = time.time()

        logger.info("Processing PDF {}...".format(pdf_path))

        # Initialize chunks list
        all_chunks = []

        # Get the pdf name, and define and create assets subdirectory
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_assets_path = os.path.join(self.config.pdf_assets_dir, pdf_name)
        os.makedirs(pdf_assets_path, exist_ok=True)

        logger.info("Extracting text and tables...")

        # Open PDF with pymupdf, and loop page-by-page (starting at 1)
        with pymupdf.open(pdf_path) as pdf:

            page_count = pdf.page_count

            for page_num, page in enumerate(pdf, start=1):

                if status_update is not None:
                    status_update("Ingesting PDF {} - Extracting data from page {} of {}..."
                                  "".format(pdf_name, page_num, page_count))

                # Get the tables chunks from the page, and add them to the list
                table_chunks, table_boundaries = self.get_page_tables(pdf_assets_path, pdf_path, pdf_name, page_num,
                                                                      page)
                all_chunks.extend(table_chunks)

                # Get the drawings chunks from the page, and add them to the list
                drawing_chunks, drawing_boundaries = self.get_page_drawings(pdf_assets_path, pdf_path, pdf_name,
                                                                            page_num, page, table_boundaries)
                all_chunks.extend(drawing_chunks)

                # Compile the non-textual boundaries
                non_text_boundaries = drawing_boundaries + table_boundaries

                # Get the text chunks from the page, and add them to the list
                text_chunks = self.get_page_text(pdf_path, pdf_name, page_num, page, non_text_boundaries)
                all_chunks.extend(text_chunks)

        # Stop timer, and get duration
        duration = time.time() - start

        logger.info("Extracted {} chunks in {} seconds".format(len(all_chunks), round(duration, 3)))

        # Return the chunks
        return all_chunks
