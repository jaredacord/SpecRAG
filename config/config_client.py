from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import List

DEFAULT_CONFIG = {
    "llm": {
        "model_name": "gemini-2.5-flash",
        "temperature": "0.0",
    },
    "faiss": {
        "storage_path": "data/faiss_index",
        "embedding_model": "models/gemini-embedding-001",
        "chunk_batch_limit": "1000",
        "seconds_between_chunks_batches": "15",
    },
    "logging": {
        "logging_level": "INFO",
        "log_dir": "logs/",
    },
    "pdf_processing": {
        "chunk_size": "1024",
        "chunk_overlap": "128",
        "object_header_height": "24",
        "object_footer_height": "30",
        "table_render_dpi": "120",
        "drawing_render_dpi": "120",
        "page_header_range": "50",
        "page_footer_range": "50",
        "max_image_discontinuity": "60",
        "min_image_size": "25,25",
        "img_ext": "png",
        "pdf_assets_dir": "data/pdf_assets",
    },
    "ingestion": {
        "ingested_pdfs": "",
    },
}


class ConfigClient:
    def __init__(self, path = "config.ini"):
        self.path = Path(path)
        self.parser = configparser.ConfigParser()

        if not self.path.exists():
            self.create_default_config()

        self.load_values()


    def load_values(self):

        self.parser.read(self.path)

        # Get LLM config values
        self.llm_model_name = self.parser.get("llm", "model_name")
        self.llm_temperature = self.parser.getfloat("llm", "temperature")

        # Get FAISS config values
        self.faiss_storage_path = self.parser.get("faiss", "storage_path")
        self.embedding_model = self.parser.get("faiss", "embedding_model")
        self.chunk_batch_limit = self.parser.getint("faiss", "chunk_batch_limit")
        self.seconds_between_chunks_batches = \
            self.parser.getint("faiss", "seconds_between_chunks_batches")

        # Get logging config values
        self.logging_level = getattr(logging, self.parser.get("logging", "logging_level").upper())
        self.log_dir = self.parser.get("logging", "log_dir")

        # Get PDF processing config values
        self.chunk_size = self.parser.getint("pdf_processing", "chunk_size")
        self.chunk_overlap = self.parser.getint("pdf_processing", "chunk_overlap")
        self.object_header_height = self.parser.getint("pdf_processing", "object_header_height")
        self.object_footer_height = self.parser.getint("pdf_processing", "object_footer_height")
        self.table_render_dpi = self.parser.getint("pdf_processing", "table_render_dpi")
        self.drawing_render_dpi = self.parser.getint("pdf_processing", "drawing_render_dpi")
        self.page_header_range = self.parser.getint("pdf_processing", "page_header_range")
        self.page_footer_range = self.parser.getint("pdf_processing", "page_footer_range")
        self.max_image_discontinuity = self.parser.getint("pdf_processing", "max_image_discontinuity")
        self.min_image_size = [
            int(v)
            for v in self.parser.get("pdf_processing", "min_image_size").split(",")
        ]
        self.img_ext = self.parser.get("pdf_processing", "img_ext")
        self.pdf_assets_dir = self.parser.get("pdf_processing", "pdf_assets_dir")

        # Get the ingested PDFs
        self.ingested_pdfs = self.get_ingested_pdfs()

    def get_ingested_pdfs(self) -> List[str]:
        raw = self.parser.get("ingestion", "ingested_pdfs", fallback="")
        if not raw.strip():
            return []
        return [p.strip() for p in raw.split(",")]

    def add_ingested_pdf(self, pdf_name):

        self.ingested_pdfs.append(pdf_name)

        self.parser.set(
            "ingestion",
            "ingested_pdfs",
            ",".join(sorted(self.ingested_pdfs)),
        )

        self.write_config()

    def create_default_config(self) -> None:
        for section, values in DEFAULT_CONFIG.items():
            self.parser[section] = values

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            self.parser.write(f)

    def write_config(self) -> None:
        with self.path.open("w") as f:
            self.parser.write(f)
