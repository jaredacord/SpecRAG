import logging


class LLMConfig:
    model_name = "gemini-2.5-flash"
    temperature = 0.0

class FAISSConfig:
    storage_path = "data/faiss_index"
    embedding_model = "models/gemini-embedding-001"
    chunk_group_limit = 1000
    seconds_between_chunks_groups = 5

class LoggingConfig:
    logging_level = logging.INFO
    log_dir = "logs/"

class PDFProcessingConfig:
    chunk_size = 1024
    chunk_overlap = 128
    object_header_height = 30
    table_render_dpi = 120
    drawing_render_dpi = 120
    max_image_discontinuity = 30
    min_image_size = [10, 10]
    img_ext = 'png'
    pdf_assets_dir = "data/pdf_assets"