# -semantic-fashion-search
A semantic search engine for fashion products using Sentence Transformers and ChromaDB, with a Streamlit UI.

# Semantic Fashion Search Engine 

A semantic search engine for fashion products built with Python, Sentence Transformers, and ChromaDB. This application allows users to search for clothing based on natural language descriptions and concepts, rather than just keywords.


## Key Features

*   **Semantic Search:** Understands the *meaning* behind a user's query (e.g., "outfit for a summer wedding") to find relevant products.
*   **Two-Stage Search & Rerank:** Uses a fast Bi-Encoder for initial candidate retrieval and a more accurate Cross-Encoder for final reranking to ensure high relevance.
*   **Interactive UI:** A simple and clean user interface built with Streamlit.
*   **Data Cleaning:** Includes a data pipeline that cleans and prepares text descriptions from the source dataset.

## Tech Stack

*   **Language:** Python
*   **Web Framework:** Streamlit
*   **AI / ML:** Sentence-Transformers, PyTorch
*   **Vector Database:** ChromaDB
*   **Data Handling:** Pandas, BeautifulSoup4

---
