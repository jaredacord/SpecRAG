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

