import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from dotenv import load_dotenv
from ttkthemes import ThemedTk

load_dotenv()

import logging
logger = logging.getLogger(__name__)

class RAGGui:
    def __init__(self, config, rag):
        """
        Initialization method for the RAGGui class. This method builds and starts the gui.

        :param config: ConfigClient instance
        :param rag: RAG instance
        """

        super().__init__()

        # Initialize variables
        self.config = config
        self.rag = rag

        # Keep track of any currently running tasks
        self.current_query_thread = None
        self.query_stop_event = threading.Event()

        # Set the tkinter theme, title, and geometry
        self.root = ThemedTk(theme="arc")
        self.root.title("SpecRag")
        self.root.geometry("1000x500")

        # Build the gui, and start the mainloop
        self.build_ui()
        self.root.mainloop()

    def build_ui(self):
        """
        This method builds the tkinter gui by calling other methods that build individual sections of the gui

        :return: None
        """

        # Call methods to build individual sections of the gui
        self.build_query_frame()
        self.build_status_frame()
        self.build_results_section()
        self.build_add_pdf_frame()

        # Set the error and status callbacks
        self.rag.set_status_callback(self.update_status)
        self.rag.set_error_callback(self.display_error_dialog)

    def build_query_frame(self):
        """
        This method builds the query input section of the gui

        :return: None
        """

        # Define the frame for the query input
        query_input_frame = ttk.Frame(self.root, padding=10)
        query_input_frame.pack(fill=tk.X)

        # Add label
        ttk.Label(query_input_frame, text="Enter Query:").grid(row=0, column=0, sticky="w", padx=(0, 5))

        # Define the query variable and add the entry box
        self.query_var = tk.StringVar()
        ttk.Entry(query_input_frame, textvariable=self.query_var).grid(row=0, column=1, sticky="ew", padx=(0, 10))

        # Add submit button linked to the on_query_submit() method
        ttk.Button(query_input_frame, text="Submit", command=self.on_query_submit).grid(row=0, column=2)

        # Finally, configure the frame
        query_input_frame.columnconfigure(1, weight=1)

    def build_status_frame(self):
        """
        This method builds the status section of the gui

        :return: None
        """

        # Define the frame for the status section
        status_frame = ttk.Frame(self.root, padding=(10, 5))
        status_frame.pack(fill=tk.X)

        # Define the status variable and set the initial status
        self.status_var = tk.StringVar(value="Idle")

        # Add the status as a centered label
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="center")
        self.status_label.pack(fill=tk.X)

    def build_results_section(self):
        """
        This method builds the results section of the gui, including the filter pane

        :return: None
        """

        # Define the frame for the results section
        results_frame = ttk.Frame(self.root, padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Add the filter and response labels
        ttk.Label(results_frame, text="Filter by Spec").grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(results_frame, text="Response").grid(row=0, column=1, sticky="w", pady=(0, 5))

        # Define the spec listbox, and populate from the list of ingested pdfs
        spec_listbox = tk.Listbox(results_frame, height=10, width=25, selectmode=tk.MULTIPLE)
        spec_listbox.grid(row=1, column=0, sticky="nsw", padx=(0, 10))

        for pdf in self.config.ingested_pdfs:
            spec_listbox.insert(tk.END, pdf)

        # Define the response textbox
        response_frame = ttk.Frame(results_frame)
        response_frame.grid(row=1, column=1, sticky="nsew")

        # Define the response text, and scrollbar
        response_text = tk.Text(response_frame, wrap=tk.WORD, height=10)
        response_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        response_scrollbar = ttk.Scrollbar(response_frame, orient=tk.VERTICAL, command=response_text.yview)
        response_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        response_text.configure(yscrollcommand=response_scrollbar.set)

        # Configure the frames
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(1, weight=1)

        # Store the listbox and textbox for later use
        self.spec_listbox = spec_listbox
        self.response_text = response_text

    def build_add_pdf_frame(self):
        """
        This method builds the frame for adding PDFs to the RAG

        :return: None
        """

        # Define the frame for the add pdf section
        add_pdf_frame = ttk.Frame(self.root, padding=10)
        add_pdf_frame.pack(fill=tk.X)

        # Add label
        ttk.Label(add_pdf_frame, text="Add PDF:").grid(row=0, column=0, sticky="w", padx=(0, 5))

        # Define the pdf path variable and add the entry box
        self.pdf_path_var = tk.StringVar()
        pdf_entry = ttk.Entry(add_pdf_frame, textvariable=self.pdf_path_var)
        pdf_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))

        # Add browse and submit buttons
        ttk.Button(add_pdf_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(add_pdf_frame, text="Submit", command=self.on_add_pdf_submit).grid(row=0, column=3)

        # Configure the frame for resizing
        add_pdf_frame.columnconfigure(1, weight=1)

    def browse_file(self):
        """
        Helper method for opening a file dialog to select a PDF file. This method does not return anything, but rather
        sets self.pdf_path_var

        :return: None
        """

        # Open file dialog in current working directory
        file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select PDF",
                                               filetypes=[("PDF files", "*.pdf")])

        # Set the pdf path variable if file was selected
        if file_path:
            self.pdf_path_var.set(file_path)

    def update_status(self, message):
        """
        Callback method for updating the status label

        :param message: Message to display, as string
        :return: None
        """

        # Set message and update any idle tasks
        self.status_var.set(message)
        self.root.update_idletasks()

    def display_error_dialog(self, message):
        """
        Callback method for displaying an error dialog

        :param message: Message to display, as string
        :return: None
        """
        messagebox.showerror("Error", message)
        self.update_status("Idle")

    def set_widget_text(self, widget, text):
        """
        Helper method for setting the text of a tkinter widget (E.g., the response text)

        :param widget: Tkinter widget
        :param text: Text to set, as string
        :return: None
        """

        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)

    def on_query_submit(self):
        """
        Method to handle query submission

        :return: None
        """

        if self.current_query_thread is not None and self.current_query_thread.is_alive():

            user_decision = messagebox.askyesno(
                "Query Already Running",
                "A query is already being processed.\n\n"
                "Do you want to stop the current query and process the new one?\n\n"
                "Click 'Yes' to stop the first query and process the second.\n"
                "Click 'No' to abort the second query."
            )

            if not user_decision:
                return
            else:
                self.stop_current_query()

        # Start the task in a new thread
        self.query_stop_event.clear()
        self.current_query_thread = threading.Thread(target=self.run_query, daemon=True)
        self.current_query_thread.start()

    def stop_current_query(self):

        if self.current_query_thread is not None and self.current_query_thread.is_alive():
            self.update_status("Stopping current query gracefully...")
            self.query_stop_event.set()
            self.current_query_thread.join()
            self.current_query_thread = None
            self.update_status("Query Cancelled")

    def run_query(self):
        """
        Method to handle the actual query task, to be run in a separate thread.

        :return: None
        """

        # Get the query text and selected specs from the gui
        query_text = self.query_var.get().strip()
        selected_indices = self.spec_listbox.curselection()
        selected_pdfs = [self.spec_listbox.get(i) for i in selected_indices]

        if self.query_stop_event.is_set():
            return

        # Return early if query is empty
        if not query_text:
            self.update_status("Idle")
            return

        # Define the metadata filter using the selected specs. If no specs were selected, pass in None
        # (denoting no filter)
        metadata_filter = None
        if selected_pdfs:
            metadata_filter = {"pdf_name": {"$in": selected_pdfs}}

        if self.query_stop_event.is_set():
            return

        # Get response, and display the response text
        response, expansive_queries, contexts = self.rag.get_context_and_answer_rrf(query_text, metadata_filter,
                                                                                    stop_event=self.query_stop_event)

        if response is not None:
            self.set_widget_text(self.response_text, response)

        self.current_query_thread = None

    def on_add_pdf_submit(self):
        """
        Method to handle add pdf submission

        :return: None
        """

        # Get the pdf path from the pdf path variable
        pdf_path = self.pdf_path_var.get().strip()

        # Display error and return in case of empty path
        if not pdf_path:
            self.display_error_dialog("PDF path not given")
            return

        # Prompt user to confirm
        confirmation_message = ("This action will add the selected PDF to the RAG. The process will take several "
                                "minutes, and use a substantial number of LLM tokens. Before proceeding, you must "
                                "have a paid LLM API key, and understand the costs. Continue?")
        result = messagebox.askyesno("Confirm Add PDF", confirmation_message)

        # Return early if user does not confirm
        if not result:
            return

        # Add pdf to RAG
        self.rag.ingest_pdfs(pdf_path)

        # Repopulate the spec listbox
        self.spec_listbox.delete(0, 'end')
        for pdf in self.config.ingested_pdfs:
            self.spec_listbox.insert(tk.END, pdf)
