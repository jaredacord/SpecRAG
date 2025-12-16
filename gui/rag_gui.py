import json
import os
import re
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
from ttkthemes import ThemedTk
from tkinter.scrolledtext import ScrolledText

from dotenv import load_dotenv

from config.config_client import ConfigClient
from rag.rag import RAG

load_dotenv()

from src.pdf_processor import PDFProcessor

import logging
logger = logging.getLogger(__name__)

class RAGGui:
    def __init__(self, config, rag):
        super().__init__()
        self.config = config
        self.rag = rag

        self.root = ThemedTk(theme="arc")
        self.root.title("SpecRag")
        self.root.geometry("1000x500")

        self.build_ui()

        self.root.mainloop()

    def build_ui(self):
        self.build_query_frame()
        self.build_status_frame()
        self.build_results_section()
        self.build_add_pdf_frame()

    def build_query_frame(self):

        query_input_frame = ttk.Frame(self.root, padding=10)
        query_input_frame.pack(fill=tk.X)

        ttk.Label(query_input_frame, text="Enter Query:").grid(row=0, column=0, sticky="w", padx=(0, 5))

        self.query_var = tk.StringVar()
        ttk.Entry(query_input_frame, textvariable=self.query_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 10)
        )

        ttk.Button(query_input_frame, text="Submit", command=self.on_query_submit).grid(
            row=0, column=2
        )

        query_input_frame.columnconfigure(1, weight=1)

    def build_status_frame(self):
        status_frame = ttk.Frame(self.root, padding=(10, 5))
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Idle")

        # Centered label
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="center"
        )
        self.status_label.pack(fill=tk.X)

        self.rag.set_status_update(self.update_status)

    def build_results_section(self):

        # Container for the whole section
        results_frame = ttk.Frame(self.root, padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Labels
        ttk.Label(results_frame, text="PDF Filters").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Label(results_frame, text="Answer").grid(
            row=0, column=1, sticky="w", pady=(0, 5)
        )

        # ---- PDF List (left) ----
        pdf_listbox = tk.Listbox(
            results_frame,
            height=10,
            selectmode=tk.MULTIPLE
        )
        pdf_listbox.grid(
            row=1, column=0, sticky="nsw", padx=(0, 10)
        )

        for pdf in self.config.ingested_pdfs:
            pdf_listbox.insert(tk.END, pdf)

        # ---- Answer Textbox (right) ----
        answer_frame = ttk.Frame(results_frame)
        answer_frame.grid(row=1, column=1, sticky="nsew")

        answer_text = tk.Text(
            answer_frame,
            wrap=tk.WORD,
            height=10
        )
        answer_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        answer_scrollbar = ttk.Scrollbar(
            answer_frame,
            orient=tk.VERTICAL,
            command=answer_text.yview
        )
        answer_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        answer_text.configure(yscrollcommand=answer_scrollbar.set)

        # Make layout responsive
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(1, weight=1)

        self.pdf_listbox = pdf_listbox
        self.answer_text = answer_text

    def build_add_pdf_frame(self):
        # --- Container frame ---
        add_pdf_frame = ttk.Frame(self.root, padding=10)
        add_pdf_frame.pack(fill=tk.X)

        # --- Label ---
        ttk.Label(add_pdf_frame, text="Add PDF:").grid(row=0, column=0, sticky="w", padx=(0, 5))

        # --- File path entry ---
        self.pdf_path_var = tk.StringVar()
        pdf_entry = ttk.Entry(add_pdf_frame, textvariable=self.pdf_path_var)
        pdf_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        ttk.Button(add_pdf_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(0, 5))

        # --- Submit button ---
        ttk.Button(add_pdf_frame, text="Submit", command=self.on_add_pdf_submit).grid(row=0, column=3)

        # Make entry expand when window resizes
        add_pdf_frame.columnconfigure(1, weight=1)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf")],
        )
        if file_path:
            self.pdf_path_var.set(file_path)

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def set_widget_text(self, widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    def on_add_pdf_submit(self):
        """
        Called when the 'Submit' button in the Add PDF frame is pressed.
        Currently, it just prints the selected PDF path.
        """
        pdf_path = self.pdf_path_var.get().strip()

        if not pdf_path:
            return

        self.rag.ingest_pdfs(pdf_path)

        self.pdf_listbox.delete(0, 'end')

        for pdf in self.config.ingested_pdfs:
            self.pdf_listbox.insert(tk.END, pdf)

    def on_query_submit(self):

        query_text = self.query_var.get().strip()

        selected_indices = self.pdf_listbox.curselection()
        selected_pdfs = [self.pdf_listbox.get(i) for i in selected_indices]

        if not query_text:
            return

        metadata_filter = None
        if selected_pdfs:
            metadata_filter = {"pdf_name": {"$in": selected_pdfs}}

        response, expansive_queries, contexts = self.rag.get_context_and_answer_rrf(query_text, metadata_filter)

        self.set_widget_text(self.answer_text, response)
