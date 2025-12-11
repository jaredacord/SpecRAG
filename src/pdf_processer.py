import logging
import os
import time
import uuid

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PDFProcessingConfig
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

class PDFProcesser:

    def __init__(self):

        # Initialize LLM client
        self.llm_client = LLMClient()

        # Get relevant config values
        self.chunk_overlap = PDFProcessingConfig.chunk_overlap
        self.chunk_size = PDFProcessingConfig.chunk_size
        self.drawing_render_dpi = PDFProcessingConfig.drawing_render_dpi
        self.img_ext = PDFProcessingConfig.img_ext
        self.min_img_x, self.min_img_y = PDFProcessingConfig.min_image_size
        self.object_header_height = PDFProcessingConfig.object_header_height
        self.object_footer_height = PDFProcessingConfig.object_footer_height
        self.page_header_range = PDFProcessingConfig.page_header_range
        self.page_footer_range = PDFProcessingConfig.page_footer_range
        self.table_render_dpi = PDFProcessingConfig.table_render_dpi
        self.max_image_discontinuity = PDFProcessingConfig.max_image_discontinuity

        # Define the text splitter, used in chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def boxes_touch(self, b1, b2, thresh):
        x0a, y0a, x1a, y1a = b1
        x0b, y0b, x1b, y1b = b2

        # Expand boxes by threshold to test adjacency
        x0a -= thresh;
        y0a -= thresh;
        x1a += thresh;
        y1a += thresh
        x0b -= thresh;
        y0b -= thresh;
        x1b += thresh;
        y1b += thresh

        # Overlap test on expanded boxes
        return not (x1a < x0b or x1b < x0a or y1a < y0b or y1b < y0a)

    def merge_two(self, b1, b2):
        return [
            min(b1[0], b2[0]),
            min(b1[1], b2[1]),
            max(b1[2], b2[2]),
            max(b1[3], b2[3]),
        ]

    def merge_boxes_iterative(self, boxes, thresh):
        boxes = boxes[:]  # shallow copy
        changed = True

        while changed:
            changed = False
            merged = []
            used = [False] * len(boxes)

            for i in range(len(boxes)):
                if used[i]:
                    continue

                current = boxes[i]

                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue

                    if self.boxes_touch(current, boxes[j], thresh):
                        current = self.merge_two(current, boxes[j])
                        used[j] = True
                        changed = True

                used[i] = True
                merged.append(current)

            boxes = merged

        return boxes

    def remove_itersection(self, boundaries, boundaries_to_remove):
        """
        This method removes all boundaries from 'boundaries' which overlap with any of the boundaries in
        'boundaries_to_remove'.

        :param boundaries: Initial list of boundaries, as list of [x0, y0, x1, y1] coordinates
        :param boundaries_to_remove: List of boundaries to remove, as list of [x0, y0, x1, y1] coordinates
        :return: List of boundaries, after removing intersections
        """

        boundaries_to_return = []

        for boundary in boundaries:
            overlap_found = False
            for boundary_to_remove in boundaries_to_remove:
                if self.boxes_touch(boundary, boundary_to_remove, 0):
                    overlap_found = True
                    break
            if not overlap_found:
                boundaries_to_return.append(boundary)

        return boundaries_to_return

    def is_table_valid(self, table, page):
        x0, y0, x1, y1 = table.bbox
        area = pymupdf.Rect(x0, y0, x1, y1)

        width = x1 - x0
        height = y1 - y0

        # If table contains no content, reject it
        if not (page.get_text("text", clip=area) or "").strip():
            return False

        # Table must have at least 2 columns non-empty columns
        non_empty_columns = [name for name in table.header.names if name is not None and name.strip()!=""]
        if table.col_count is None or len(non_empty_columns)<2:
            return False

        # If table is too large, reject it
        if height > page.rect.height * 0.9:
            return False

        # Reject tables with no extracted text
        if table.extract() is None or not table.extract():
            return False

        # If table is within page header or footer, reject it
        if y0 < self.page_header_range or y1 > (page.rect[3] - self.page_footer_range):
            return False

        return True

    def is_drawing_valid(self, drawing, page):

        drawing_coordinates = drawing['rect']
        x0, y0, x1, y1 = drawing_coordinates[0:4]
        area = pymupdf.Rect(x0, y0, x1, y1)

        width = x1 - x0
        height = y1 - y0

        # If drawing is an implementation note, reject it (PCI spec specific)
        if page.get_text("text", clip=area).strip().lower().startswith("implementation note"):
            return False

        # If drawing is too small, reject it
        if width < self.min_img_x or height < self.min_img_y:
            return False

        # If drawing is too large, reject it
        if height > page.rect.height * 0.9:
            return False

        # If table is within page header or footer, reject it
        if y0 < self.page_header_range or y1 > (page.rect[3] - self.page_footer_range):
            return False

        return True

    def get_image_description(self, image_path):
        """
        This method uses the LLM client to describe the image at the given path, is about self.chunk_size characters.

        :param image_path: image path, as string
        :return: LLM generated description of the image, as string
        """

        return "Sample Output"

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
                    "type": "table",
                    "source": <path to PDF file as str>,
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
            if not self.is_table_valid(table, page):
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

            # Define the boundaries for the table, header, and table + header
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

        # Return chunks, and table boundaries
        return chunks_to_return, table_boundaries

    def get_page_drawings(self, pdf_assets_path, pdf_path, pdf_name, page_num, page, table_boundaries):
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

        :param pdf_assets_path: Path to directory where drawings will be saved
        :param pdf_path: Path to PDF (used in metadata)
        :param pdf_name: Name of PDF (used in file naming)
        :param page_num: Doc page number (used in metadata), starting from 1
        :param page: Page object from pymupdf
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

            if not self.is_drawing_valid(drawing, page):
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

        drawing_boundaries = self.remove_itersection(drawing_boundaries, table_boundaries)

        drawing_boundaries = self.merge_boxes_iterative(drawing_boundaries, self.max_image_discontinuity)

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
                        "page": page_num,
                        "drawing_num": drawing_num,
                        "drawing_path": drawing_output_path
                    }
                }
            )

            drawing_num += 1

        # Return chunks, and drawing boundaries
        return chunks_to_return, drawing_boundaries

    def get_page_text(self, pdf_path, page_num, page, non_text_boundaries):
        """
        Method to extract text from a PDF page. Since the page content includes drawings, tables, and so on with
        textual aspects, we first filter out the non-textual portions of the page (since the information is already
        encoded). We then extract the remaining pure textual content, and split it into chunks.
        One assumption we make is that the non-textual portions of the page fill the page horizontally.
        Each text chunk is given as:
            {
                "page_content": <extracted text>,
                "metadata": {
                    "type": "text",
                    "source": <path to PDF file as str>,
                    "page": <page number as int>,
                    "chunk_index": <index of chunk within page as int>
                }
            }

        :param pdf_path: Path to PDF (used in metadata)
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
                        "page": page_num,
                        "chunk_index": i
                    }
                }
            )

        # Return chunks
        return chunks_to_return

    def pdf_to_chunks(self, pdf_path):
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
        pdf_assets_path = os.path.join(PDFProcessingConfig.pdf_assets_dir, pdf_name)
        os.makedirs(pdf_assets_path, exist_ok=True)

        logger.info("Extracting text and tables...")

        # Open PDF with pymupdf, and loop page-by-page (starting at 1)
        with pymupdf.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):

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
                text_chunks = self.get_page_text(pdf_path, page_num, page, non_text_boundaries)
                all_chunks.extend(text_chunks)

        # Stop timer, and get duration
        duration = time.time() - start

        logger.info("Extracted {} chunks in {} seconds".format(len(all_chunks), round(duration, 3)))

        # Return the chunks
        return all_chunks
