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
- [Future Enhancements](#future-enhancements)

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

2. **(Optional) Setup Virtual Environment**
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

5. **(Optional) Copy over faiss_index and pdf_assets folders**
   - If you have pre-existing `faiss_index` and accompanying `pdf_assets` folders:
     - Copy the faiss index into `/data/faiss_index/`
     - Copy the pdf assets into `/data/pdf_assets/`. 
     - Copy the original pdfs into `/data/pdfs/` (optional).
     - Populate the `ingested_pdfs` field in config.ini with the pdf filenames, or copy config.ini into the root 
     directory.

6. **Run the Application**
   - To run the application:
     ```bash
     python app.py
     ```

---

## Usage

Since PDF ingestion takes a long time and can be costly, the preferred usage is to generate the `data/` folder (which
includes `data/faiss_index/`, `data/pdf_assets/`, and `data/pdfs/` (optional)) using a list of commonly used 
specifications once, and distribute to other users as needed.

Once setup and instillation are complete, navigate to the root directory of the project and run `python app.py`. 
After a few seconds, the GUI will open.

### Description of GUI Elements

![usage_1.png](/assets/usage_1.png)

1. **Query Input**: Enter your query here. Click submit once query and filter (see below) are set.
2. **Status Indicator**: Displays the status of any running processes. For example, when generating a response to a 
query, the status indicator will move from 'Expanding Query' to 'Retrieving Chunks for {x} queries' to 'Answering Query 
using {y} chunks' to 'Done'. 
3. **Spec Filter**: Multi-selection list for filtering which specifications to use for the query. The filter is applied 
during the retrieval step, so a response will need to be re-generated for a given query if the filter is changed. If no
specifications are selected, all specifications will be used. Note: the list shown above is an example. The actual list 
of specifications will reflect the specifications that have been ingested.
4. **Response Pane**: Displays the response to the query. 
5. **PDF Ingestion Input**: If you want to add a new pdf to the RAG, enter the path to the pdf here. After clicking 
submit, a confirmation window will appear. Ensure you are using a paid Google API key if you use this feature. Also, 
given the current implementation of SpecRAG, the window may freeze during pdf ingestion, rendering the status indicator 
incorrect. If this occurs, watch the terminal window used when running the application for updates.

### Examples

Here is a simple example query, filter, and response:

![example_1.png](/assets/example_1.png)

Here are some more examples highlighting different processing paths. For brevity, the image is omitted for each, and 
instead the query, filter, and response are shown as text.

1. **Another example**
   - Query: "Where can I find information about the nvme-mi version the device complies to?"
   - Filter: NVMe-MI-1.4c
   - Response:
   ```
    Information about the NVMe-MI version a device complies to can be found in the NVM Subsystem Information Data Structure (NVMe-MI-1.2c, page 95).

    Specifically:
   *   **NVMe-MI Major Version Number (MJR)**: This field is located at byte 01 and indicates the major version number of the specification (NVMe-MI-1.2c, page 95). It shall be set to 1h (NVMe-MI-1.2c, page 95).
   *   **NVMe-MI Minor Version Number (MNR)**: This field is located at byte 02 and indicates the minor version number of the specification (NVMe-MI-1.2c, page 95). It shall be set to 2h (NVMe-MI-1.2c, page 95).

    To retrieve this data structure, the Data Structure Type (DTYP) field in the Read NVMe-MI Data Structure – NVMe Management Dword 0 (bits 31:24) should be set to `00h` for "NVM Subsystem Information" (NVMe-MI-1.2c, page 94).
   ```

2. **An unrelated query.**
   - Query: "What is the current weather in Berlin?" 
   - Filter: None  
   - Response: 
   ```
   The provided context does not contain any information about the current weather in Berlin. Therefore, 
   I cannot answer your question.
   ```

3. **Interpreting information from a flow chart image.** Note, the following response is based on *Figure 3-19 Receive 
   Data Link Layer Handling of TLPs*, found on page 241 of the PCIe 5.0 specification, in conjunction with textual 
   information found in the same section.
    - Query: "When handling TLPs in the data link layer, will an incorrect CRC value mark the TLP as a bad TLP?"
    - Filter: PCI_Express_5.0_v1
    - Response:
    ```
    Yes, an incorrect LCRC value can mark a TLP as a Bad TLP in the Data Link Layer (PCI_Express_5.0_v1, page 239, 241).

    The Data Link Layer checks the 32-bit LCRC value of a received TLP by applying the same algorithm used for calculation to the received TLP (excluding its 32-bit LCRC field) and comparing the calculated result with the value in the LCRC field of the received TLP (PCI_Express_5.0_v1, page 239).

    A TLP is considered a "Bad TLP" under the following conditions related to its LCRC:
    *   If the TLP was *not* indicated as nullified by the Physical Layer, and the calculated LCRC is *not* equal to the received LCRC value (PCI_Express_5.0_v1, page 241).
    *   If the Physical Layer reports that the received TLP was nullified, but the LCRC does *not* match the logical NOT of the calculated value, the TLP is corrupt and this is considered a Bad TLP error (PCI_Express_5.0_v1, page 239, 241).

    When a TLP is identified as a Bad TLP error, it is discarded, any allocated storage for it is freed, and if the NAK_SCHEDULED flag is clear, a Nak DLLP is scheduled for transmission immediately and the NAK_SCHEDULED flag is set (PCI_Express_5.0_v1, page 239, 241).

    It is important to note that if the Physical Layer reports that the received TLP was nullified, and the LCRC *is* the logical NOT of the calculated value, the TLP is discarded and storage is freed, but this specific scenario is *not* considered an error (PCI_Express_5.0_v1, page 239, 241).
    ```

4. **Extracting values from a table**
   - Query: "What are the options for data type in the read nvme-mi data command? Can you provide a value-description mapping as a dict, with values given by their decimal equivalents? You can ignore the reserved fields."
   - Filter: NVMe-MI-1.4c
   - Response: 
   ```
    The options for the Data Structure Type (DTYP) field in the Read NVMe-MI Data Structure command are as follows (NVMe-MI-1.2c, page 94):

    {
      0: "NVM Subsystem Information",
      1: "Port Information",
      2: "Controller List",
      3: "Controller Information",
      4: "Optionally Supported Command List",
      5: "Management Endpoint Buffer Command Support List"
    }
   ```

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

## Future Enhancements
