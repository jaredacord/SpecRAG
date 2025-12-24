# SpecRAG

SpecRAG is a Python-based Retrieval-Augmented Generation (RAG) application designed to make dense NVMe and PCIe 
specifications more accessible. Rather than manually navigating hundreds of pages of highly technical PDFs, SpecRAG 
enables users to ask natural-language questions and receive grounded, specification-backed answers.

The application ingests official NVMe and PCIe specification PDFs, chunks and embeds their contents, and indexes them 
in a FAISS vector store. At query time, SpecRAG expands the user’s question via query expansion, retrieves relevant 
context for each generated query, and re-ranks the combined results using Reciprocal Rank Fusion (RRF). The top-ranked 
context chunks are then provided to a large language model alongside the original query to generate a focused, 
spec-aware response.

SpecRAG is intended as a first step in researching NVMe and PCIe specifications—helping engineers, students, and 
researchers quickly locate and understand relevant sections while preserving traceability back to the original documents.

---
Jump to:

- [Theory of Operation](#theory-of-operation)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Theory of Operation

### Specification Ingestion 

![ingestion_pipeline.png](/assets/ingestion_pipeline.png)

When a PDF is ingested, it is first split into pages, and each page is processed separately. For each page, SpecRAG 
extracts tables, drawings, then body text, in that order (implicitly giving priority to table data), using the 
PyMuPDF library for data extraction. Each table and drawing is stored locally as an image file, to be used later during 
context retrieval. 

Body text is split into chunk contents using langchain's RecursiveCharacterTextSplitter. Table data is 
extracted as plain-text, and split into chunk contents using the same method. For drawings, an LLM-generated technical 
description is used as a single chunk content. Each chunk content is then augmented with metadata, defined as so:

**Table**
```
    "metadata":{
        "uuid": <generated uuid4 string>,
        "type": "table",
        "source": <path to PDF file as str>,
        "pdf_name": <pdf_name as str>,
        "page": <page number as int>,
        "table_num": <table number by page as int>,
        "table_path": <path to saved image file>,
        "chunk_index": <index of chunk within table as int>
```

**Drawing**
```
    "metadata": {
        "uuid": <generated uuid4 string>,
        "type": "drawing",
        "source": <path to PDF file as str>,
        "pdf_name": <pdf_name as str>,
        "page": <page number as int>,
        "drawing_num": <drawing number by page as int>,
        "drawing_path": <path to saved image file>
    }
```

**Text**
```
    "metadata": {
        "uuid": <generated uuid4 string>,
        "type": "text",
        "source": <path to PDF file as str>,
        "pdf_name": <pdf_name as str>,
        "page": <page number as int>,
        "chunk_index": <index of chunk within page as int>
    }
```

The chunks (with each chunk composed of the chunk content and chunk metadata) for each page are then aggregated into a 
list of chunks for the given PDF. Once the PDF is rendered as list of chunks, the chunks are then batched and indexed 
into a FAISS vector store.

### Query Processing

![query_pipeline.png](/assets/query_pipeline.png)

Each query is expanded by generating 3-5 similar queries. N (config.ini `top_n_retrieval`) context chunks for each 
query (including the original) are then retrieved from the FAISS vector store using a similarity search. These lists 
of context chunks are then aggregated into a single list using Reciprocal Rank Fusion (RRF), with no priority given to 
the original query (since specs contain a lot of jargon). This list is truncated to the top m (config.ini 
`top_n_context_rrf`) chunks.

For each context chunk in the RRF list, the associated asset (table, drawing) is retrieved from the local storage if
applicable for that chunk type. The content of these context chunks, or associated asset if applicable, is then 
compiled as context. The query is then augmented with this context, and sent to the multi-modal LLM for response 
generation.

---

## Installation and Setup

### Prerequisites

1. **Python**: Ensure you have Python 3.11 installed.
2. **Google API Key**: SpecRAG uses Gemini-2.5-flash, so you will need a Google API key. The free tier should be enough 
for light query usage, however, you should consider upgrading to a paid tier for PDF ingestion.  

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd SpecRAG
   ```

2. **Setup Environment**
   - Create a virtual environment:
     ```bash
     python -m venv .venv
     ```
   - Activate the virtual environment:
     - For Linux/Mac:
       ```bash
       source .venv/bin/activate
       ```
     - For Windows:
       ```bash
       .venv\Scripts\activate
       ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   - Copy the `.env.sample` file to `.env`:
     ```bash
     cp .env.sample .env
     ```
   - Add your Google API key in the `.env` file:
     ```dotenv
     GOOGLE_API_KEY=your_google_api_key_here
     ```

5. **Run the Application**
   - To run the main application:
     ```bash
     python main2.py
     ```
   - Use the GUI (if required) by running:
     ```bash
     python gui/rag_gui.py
     ```

---

## Usage

---

## Project Structure

|       Folder        |         File          |                          Description                          |
|:-------------------:|:---------------------:|:-------------------------------------------------------------:|
|      `assets/`      |           *           |              Contains assets used in the README               |
|      `config/`      |   config_client.py    | Contains logic for reading and writing project configurations |
| `data/faiss_index/` |           *           |                    Stores the FAISS store                     |
| `data/pdf_assets/`  |           *           |   Stores assets (tables, drawings) extracted from each pdf    |
|    `data/pdfs/`     |           *           |              Stores pdfs that have been ingested              |
|       `gui/`        |      rag_gui.py       |                 Contains the gui for SpecRAG                  |
|       `logs/`       |           *           |                  Stores logs for the project                  |
|       `rag/`        |        rag.py         |        Contains the ingestion and query pipeline logic        |
|       `src/`        |    faiss_client.py    |    Contains the logic for interacting with the FAISS store    |
|       `src/`        |     llm_client.py     |             Contains the logic for all LLM calls              |
|       `src/`        |   pdf_processor.py    |       Contains the logic for parsing and chunking pdfs        |
|      `utils/`       | pdf_processor_util.py |        Contains utility functions for pdf_processor.py        |
|      `utils/`       |     rag_utils.py      |             Contains utility functions for rag.py             |
|         `/`         |      .env.sample      |      Sample env file, for defining environment variables      |
|         `/`         |        app.py         |                    Entry point for SpecRAG                    |
|         `/`         |      config.ini       |                      Configuration file                       |
|         `/`         |   requirements.txt    |                  Lists project dependencies                   |


---

