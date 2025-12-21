import pymupdf

class PDFProcessorUtils:
    def __init__(self, config):
        """
        Initialization method for the PDFProcessorUtils utility class.

        :param config: ConfigClient instance
        """
        
        self.config = config

        # Get relevant config values
        self.min_img_x, self.min_img_y = self.config.min_image_size
        self.page_header_range = self.config.page_header_range
        self.page_footer_range = self.config.page_footer_range
        self.table_render_dpi = self.config.table_render_dpi
        self.max_image_discontinuity = self.config.max_image_discontinuity

    def boxes_touch(self, b1, b2, thresh):
        """
        Checks if two boxes (defined by x0, y0, x1, y1) overlap by a certain threshold

        :param b1: Box 1, as [x0, y0, x1, y1] coordinates
        :param b2: Box 2, as [x0, y0, x1, y1] coordinates
        :param thresh: Overlap threshold
        :return: True if boxes overlap, False otherwise
        """

        # Unpack box coordinates
        x0a, y0a, x1a, y1a = b1
        x0b, y0b, x1b, y1b = b2

        # Expand boxes by threshold
        x0a -= thresh;
        y0a -= thresh;
        x1a += thresh;
        y1a += thresh
        x0b -= thresh;
        y0b -= thresh;
        x1b += thresh;
        y1b += thresh

        # Return if they overlap or not
        return not (x1a < x0b or x1b < x0a or y1a < y0b or y1b < y0a)

    def merge_two_boxes(self, b1, b2):
        """
        Method which merges two boxes into one

        :param b1: Box 1, as [x0, y0, x1, y1] coordinates
        :param b2: Box 2, as [x0, y0, x1, y1] coordinates
        :return: Merged box area, as [x0, y0, x1, y1] coordinates
        """

        return [
            min(b1[0], b2[0]),
            min(b1[1], b2[1]),
            max(b1[2], b2[2]),
            max(b1[3], b2[3]),
        ]

    def merge_boxes_iterative(self, boxes, thresh):
        """
        Method which merges all boxes in a list that overlap by a certain threshold

        :param boxes: List of boxes, as [x0, y0, x1, y1] coordinates
        :param thresh: Overlap threshold
        :return: Merged boxes, as list of [x0, y0, x1, y1] coordinates
        """

        # Create a shallow copy of the boxes
        boxes = boxes[:]


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
                        current = self.merge_two_boxes(current, boxes[j])
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
        """
        Method to check if the extracted table is a valid table

        :param table: PyMuPdf table object
        :param page: PyMuPdf page object
        :return: True if table is valid, False otherwise
        """

        # Extract table coordinates, and create Rect area object
        x0, y0, x1, y1 = table.bbox
        area = pymupdf.Rect(x0, y0, x1, y1)

        # If table contains no content, reject it
        if not (page.get_text("text", clip=area) or "").strip():
            return False

        # Table must have at least 2 columns non-empty columns
        non_empty_columns = [name for name in table.header.names if name is not None and name.strip()!=""]
        if table.col_count is None or len(non_empty_columns)<2:
            return False

        # Reject tables with no extracted text
        if table.extract() is None or not table.extract():
            return False

        # If table is within page header or footer, reject it
        if y0 < self.page_header_range or y1 > (page.rect[3] - self.page_footer_range):
            return False

        return True

    def is_drawing_valid(self, drawing, page):
        """
        Method to check if the extracted drawing is a valid drawing

        :param drawing: PyMuPdf drawing object
        :param page: PyMuPdf page object
        :return: True if drawing is valid, False otherwise
        """

        # Extract drawing coordinates, and create Rect area object
        drawing_coordinates = drawing['rect']
        x0, y0, x1, y1 = drawing_coordinates[0:4]
        area = pymupdf.Rect(x0, y0, x1, y1)

        # Get drawing dimensions
        width = x1 - x0
        height = y1 - y0

        # If drawing is an implementation note, reject it (PCI spec specific)
        if page.get_text("text", clip=area).strip().lower().startswith("implementation note"):
            return False

        # If drawing is too small, reject it
        if width < self.min_img_x or height < self.min_img_y:
            return False

        # If table is within page header or footer, reject it
        if y0 < self.page_header_range or y1 > (page.rect[3] - self.page_footer_range):
            return False

        return True