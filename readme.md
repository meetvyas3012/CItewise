# Multi-Document Research Assistant

A powerful Retrieval-Augmented Generation (RAG) system that enables users to upload and query multiple documents simultaneously. This project provides AI-generated, citation-supported answers powered by advanced semantic and keyword retrieval, embedding models, and LLMs. It supports various document formats and delivers transparent, traceable AI research assistance.

## Features

- **Multi-format Document Support:** Upload PDFs, DOCX, TXT, HTML, and Markdown files with robust text extraction including OCR fallback for scanned PDFs.
- **Advanced Document Processing:** Multi-layer text extraction, cleaning, and semantic chunking to preserve context and optimize retrieval efficiency.
- **Embedding & Vector Search:** Utilizes Sentence-Transformers to create semantic embeddings and FAISS for highly efficient similarity search.
- **Hybrid Retrieval System:** Combines BM25 keyword search with vector similarity search using Reciprocal Rank Fusion for superior result ranking.
- **AI-powered Response Generation:** Generates comprehensive, cited answers using local (GPT4All) or cloud-based (OpenAI) language models with confidence scoring.
- **Interactive Web Interface:** Built on Streamlit for easy document management, query input, and rich result visualization including citation traceability.
- **Modular & Scalable Architecture:** Supports offline deployment and cloud API integration while ensuring privacy and customization.

## Tech Stack & Key Libraries

Python, Streamlit, GPT4All, OpenAI API, FAISS, Sentence-Transformers, SQLAlchemy, pdfplumber, pytesseract, python-docx, BeautifulSoup, rank-bm25, numpy, pandas, pydantic, loguru, scikit-learn, nltk, plotly

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Create a Python virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables for API keys (if using OpenAI):

   Create a `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

- Upload your documents via the web interface.
- Once processed, ask natural language questions across all uploaded documents.
- View AI-generated answers with inline citations linking back to source documents.
- Explore document library and metadata within the interface.
- Use confidence scores and citation details to verify answer reliability.

## Project Structure

- `core/` – Core modules: document processing, embeddings, search, generation
- `config/` – Configuration and settings
- `data/` – Processed documents, embeddings, and indexes storage
- `streamlit_app.py` – Main Streamlit web application entry point
- `requirements.txt` – Dependencies list

## Contribution

Contributions are welcome! Please fork the repository, create a branch, and submit a pull request.


## Acknowledgments

- Uses [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search.
- Utilizes [Sentence-Transformers](https://www.sbert.net/) for embedding generation.
- Powered by Streamlit for interactive web app UI.
- Supports GPT4All and OpenAI GPT models for LLM-based response generation.

Feel free to customize this README further according to your repository name and specifics. Let me know if you want me to generate a markdown file content ready to copy-paste!
