# 🩺 Med_bot - Medical RAG QA Chatbot

An open-source, local **Retrieval-Augmented Generation (RAG)** Medical Assistant chatbot powered by **LangChain**, **FAISS**, **HuggingFace Embeddings**, **Llama 2**, and **Chainlit**.

This application allows users to ingest custom medical PDF reference materials, index them locally into a vector store, and perform interactive, context-aware Q&A directly through a modern web user interface with automated source citation.

---

## 📌 Features

- **Document Ingestion**: Extracts text from medical PDF files stored in `data/` and chunks them for semantic indexing.
- **Vector Search**: Uses `FAISS` and `sentence-transformers/all-MiniLM-L6-v2` to compute embeddings and retrieve relevant context.
- **Local LLM Inference**: Employs quantized Llama 2 (`TheBloke/Llama-2-7B-Chat-GGML`) via `CTransformers` for private, CPU-friendly answer generation without external API keys.
- **Interactive UI**: Powered by **Chainlit** for real-time streaming, chat session management, and response formatting.
- **Source Citation**: Displays source document names and page numbers alongside generated answers for verification.

---

## 🏗 Architecture & Workflow

```
[ PDF Documents ] ──> PyPDFLoader & Text Splitter
                             │
                             ▼
                 [ HuggingFace Embeddings ]
                             │
                             ▼
                    [ FAISS Vector Store ]
                             │
[ User Query ] ──> RetrievalQA Chain (Top-k Retrieval) ──> [ Llama-2 LLM ] ──> [ Chainlit UI ]
```

1. **Ingestion (`ingest.py`)**: Loads documents from `./data/`, splits them into overlapping chunks (500 characters, 50 overlap), embeds them using `all-MiniLM-L6-v2`, and saves the index to `./vectorstore/db_faiss`.
2. **Retrieval & QA (`model.py`)**: Loads the FAISS vector database, constructs a customized prompt template, queries the Llama-2 LLM with top matches, and streams the answer back to the Chainlit interface with source metadata.

---

## 📁 Repository Structure

```
Med_bot/
├── data/                    # Directory for input medical PDF files (created by user)
├── vectorstore/             # Directory where FAISS index files are saved after ingestion
│   └── db_faiss/
├── ingest.py                # Script to parse PDFs and create the FAISS vector database
├── model.py                 # Core RAG logic and Chainlit web UI runner
├── requirements.txt         # Project Python dependencies
└── README.md                # Project documentation
```

---

## 🛠 Tech Stack

- **Framework**: LangChain (`langchain`, `langchain_community`)
- **UI**: Chainlit (`chainlit`)
- **LLM**: Llama-2-7B Chat GGML (`CTransformers`)
- **Embeddings**: Sentence-Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (`faiss-cpu`)
- **Document Parser**: PyPDF (`pypdf`)

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.9+ installed on your system.
- Git (optional, for version control).

### 2. Create Virtual Environment & Install Dependencies
```bash
# Clone or open repository folder
cd Med_bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
pip install chainlit ctransformers
```

---

## 💻 Usage Instructions

### Step 1: Add Medical PDF Documents
1. Create a `data/` folder inside the project root if it does not exist:
   ```bash
   mkdir data
   ```
2. Place your target medical reference PDF documents into the `data/` folder.

### Step 2: Build the Vector Store
Run the ingestion script to process the documents and create the local vector store:
```bash
python ingest.py
```
> This will generate the vector database files under `vectorstore/db_faiss/`.

### Step 3: Run the Chatbot Interface
Launch the Chainlit web UI:
```bash
chainlit run model.py -w
```
Open your browser and navigate to `http://localhost:8000` to interact with **Med_bot**.

---

## ⚙️ Configuration & Customization

- **Prompt Engineering**: Modify `custom_prompt_template` in `model.py` to change how the bot structures its answers or handles missing context.
- **Chunking Parameters**: Adjust `chunk_size` and `chunk_overlap` in `ingest.py` to optimize document chunking based on your source materials.
- **LLM Parameters**: Fine-tune `max_new_tokens` and `temperature` in `load_llm()` inside `model.py`.

---

## ⚠️ Notes & Troubleshooting

- **CPU Performance**: The model runs on CPU using quantized GGML weights. First response loading may take a moment while the model initializes.
- **Dangerous Deserialization**: `FAISS.load_local` includes `allow_dangerous_deserialization=True` in `model.py` to load locally trusted `.pkl` vector files created by `ingest.py`. Ensure only trusted documents are ingested.
