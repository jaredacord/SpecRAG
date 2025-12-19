import time
from pathlib import Path

from dotenv import load_dotenv

from config.config_client import ConfigClient
from gui.rag_gui import RAGGui
from rag.rag import RAG

load_dotenv()

import logging
logger = logging.getLogger(__name__)

class main:

    def __init__(self):
        """
        This method initializes the main application
        """

        # Initialize the config client. This comes first, as other classes may need config values
        self.config = ConfigClient()

        # Setup logging
        self.setup_logging()

        # Initialize the RAG and RAG gui
        self.rag = RAG(self.config)
        self.rag_gui = RAGGui(self.config, self.rag)

    def setup_logging(self):
        """
        Method to setup logging

        :return: None
        """

        # Define the log path, and create the log directory if it doesn't exist
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"log_{time.strftime('%Y%m%d')}.log"

        # Setup logging
        logging.basicConfig(
            level=self.config.logging_level,
            format='%(levelname)s - %(asctime)s - %(name)s - %(message)s',
            encoding='utf-8',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # This adds console output
            ]
        )

if __name__ == '__main__':
    main()
