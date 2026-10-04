# Med_bot: Medical Retrieval-Augmented Generation (RAG) System

Med_bot is a domain-specific Retrieval-Augmented Generation (RAG) assistant designed for accurate, context-grounded medical information retrieval. Built with LangChain, FAISS, BM25, and Llama 3.1, the system allows users to query medical literature with strict guardrails against hallucination and external parametric memory leakage.

---

## Overview

Medical question-answering systems require strict adherence to reference literature to prevent misinformation. Med_bot implements an end-to-end pipeline that ingests medical textbooks and reference documentation, constructs a hybrid sparse-dense index, and executes grounded retrieval. If an inquiry cannot be answered from the provided reference documents, the system explicitly refuses rather than guessing from pre-trained model weights.

---

## Architecture and Workflow

```
[ User Query ]
      │
      ├──> [ 1. Input Validation ] (Empty/Punctuation Check)
      │
      ├──> [ 2. Query Condensation ] (Bulk History / Long Query Truncation)
      │
      ├──> [ 3. Prompt Injection Guard ] (Adversarial Jailbreak Detection)
      │
      ├──> [ 4. Emergency Red-Flag Triage ] ── (High Risk) ──> Immediate Emergency Warning
      │         (Low Risk)
      │
      ├──> [ 5. Medication & Dosage Guard ] ── (Personal Dosage) ──> Safe Refusal & Doctor Referral
      │         (Educational)
      │
      ├──> [ 6. Out-of-Domain Classifier ]  ── (Non-Medical) ──> Medical Scope Disclaimer
      │         (In-Domain)
      │
      ├──> [ 7. Pre-Retrieval Typo Normalizer ] (Medical Lexicon Normalization)
      │
      ├──> [ 8. Hybrid Ensemble Retriever ] (BM25 Lexical + FAISS Dense Semantic)
      │
      ├──> [ 9. Retrieval Confidence Gate ] ── (Irrelevant / Low Confidence) ──> Grounded Refusal
      │         (High Confidence)
      │
      └──> [ 10. Grounded LLM Inference ] (Llama 3.1 + Non-Diagnostic Disclaimers)
                │
                ▼
           [ Chainlit Web UI ]
            • Token Streaming
            • Verified Page-Level Citations: datasets.pdf (Page X)
            • Graceful Error Handling (Sanitized Fallback Messages)
```

---

## Key Capabilities & Edge Case Coverage

- **Emergency Red-Flag Triage Gate (Edge Case 1 & 3)**: Intercepts acute life-threatening symptoms (e.g., severe chest pain, shortness of breath, acute neurological signs) based on clinician-reviewed triage rules, triggering immediate emergency service advisories before model execution.
- **Normal Informational Retrieval (Edge Case 2)**: Direct, structured, and factual answers derived from reference documents without unsolicited differential sections or false emergency alerts.
- **Non-Definitive Diagnostic Boundary (Edge Cases 3 & 7)**: Adheres to clinical safety by never diagnosing ("You have X"). Mandates the non-diagnostic disclaimer: *"These symptoms may be associated with several conditions described in the reference material. MedBot cannot confirm a diagnosis."* followed by distinguishing clinical features.
- **Strict Grounding & Unknown Disease Refusal (Edge Case 4)**: Explicitly refuses to hallucinate external conditions absent from the literature.
- **Out-of-Domain Classification (Edge Case 5)**: Intercepts non-medical inquiries (politics, programming, general trivia) before retrieval.
- **Medication & Dosage Guard (Edge Case 6)**: Strictly distinguishes general pharmacology education from personalized dosage and prescription requests, safely referring patients to licensed physicians or pharmacists.
- **Pre-Retrieval Typo Normalization (Edge Case 8)**: Dedicated medical terminology normalization layer before retrieval ensures sparse keyword search (BM25) and dense embeddings accurately resolve clinical terms.
- **Retrieval Confidence Gate (Edge Case 9)**: Evaluates semantic distance and lexical overlap before invoking the LLM. If retrieval confidence is low, gracefully refuses to prevent downstream hallucination.
- **Multi-Source Conflicting Information Resolution (Edge Case 10)**: Documents are enriched with source authority and version metadata. Prompt guidelines instruct the model to explicitly surface differing viewpoints rather than silently favoring one.
- **Programmatic Prompt Injection Guard (Edge Case 11)**: Scans and blocks prompt override and jailbreak patterns before pipeline execution.
- **Input Validation & Bulk Query Handling (Edge Cases 12 & 13)**: Validates empty or punctuation-only prompts and intelligently condenses multi-page medical histories.
- **Graceful Error Handling (Edge Case 14)**: Catches runtime exceptions and displays sanitized, user-friendly notices without leaking technical stack traces.
- **Granular Page-Level Provenance (Edge Case 15)**: Retains page numbers during chunking and outputs citations referencing both document name and exact page numbers.

---

## Repository Structure

```
Med_bot/
├── data/                    # Source medical PDF files (e.g., datasets.pdf)
├── vectorstore/             # Serialized FAISS index with page-level metadata
│   └── db_faiss/
├── safety.py                # Triage policy, input validation, injection & dosage guardrails
├── ingest.py                # PDF parser preserving page numbers and source authority
├── model.py                 # Hybrid retriever, confidence gate, prompt pipeline & Chainlit UI
├── requirements.txt         # Project dependencies
├── chainlit.md              # Chainlit welcome interface configuration
└── README.md                # Technical documentation
```

---

## Tech Stack

- **Orchestration**: LangChain (`langchain`, `langchain-community`, `langchain-huggingface`)
- **Large Language Model**: Meta Llama 3.1 8B Instruct (`meta-llama/Llama-3.1-8B-Instruct`)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Dense Vector Database**: FAISS (`faiss-cpu`)
- **Sparse Lexical Search**: BM25 (`rank-bm25`)
- **Document Processing**: PyPDF (`pypdf`)
- **User Interface**: Chainlit (`chainlit`)

---

## Setup and Installation

### 1. Prerequisites
- Python 3.10 or higher
- Git
- A Hugging Face account and an API token with inference permissions

### 2. Environment Configuration
Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/Med_bot.git
cd Med_bot

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. API Token Configuration
Create a `.env` file in the root directory and add your Hugging Face API token:

```env
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here
```

---

## Ingestion and Usage

### Step 1: Ingest Medical Literature
Place your medical PDF documents into the `data/` folder, then run the ingestion script:

```bash
python ingest.py
```

This processes the PDFs, splits text while preserving section boundaries, computes vector embeddings, and serializes the index to `vectorstore/db_faiss/`.

### Step 2: Start the Web Application
Run the Chainlit interface:

```bash
chainlit run model.py -w
```

Navigate to `http://localhost:8000` in your browser to interact with the medical assistant.

---

## Medical Disclaimer

Med_bot is an academic and informational research prototype developed for context-grounded retrieval. It is not a certified diagnostic device and does not provide medical diagnoses, treatment plans, or emergency care instructions. All clinical decisions must be made by qualified healthcare professionals.
