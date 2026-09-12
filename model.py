import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    from langchain_classic.retrievers import EnsembleRetriever

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

try:
    from langchain.chains import RetrievalQA
except (ImportError, ModuleNotFoundError):
    from langchain_classic.chains import RetrievalQA

import chainlit as cl

# Path to the FAISS vector database
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Pretrained Open Source Transformer on Hugging Face
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Morphological tokenizer for BM25: handles plurals, hyphenation, and variations (e.g. rashes -> rash, headaches -> headache)
def bm25_preprocess(text: str):
    words = re.findall(r'\w+', text.lower())
    stemmed = []
    for w in words:
        if w.endswith('es') and len(w) > 3:
            stemmed.append(w[:-2])
        elif w.endswith('s') and not w.endswith('ss') and len(w) > 3:
            stemmed.append(w[:-1])
        else:
            stemmed.append(w)
    return stemmed

# Typo-tolerant, intent-aware medical prompt template with strict knowledge boundary
custom_prompt_template = """You are an expert AI Medical Assistant. Your task is to provide accurate, concise, and safe medical information based strictly and exclusively on the provided reference context.

CRITICAL INSTRUCTIONS & BOUNDARIES:
1. STRICT INFORMATION BOUNDARY (NO EXTERNAL KNOWLEDGE / NO HALLUCINATION):
   - You MUST answer questions using ONLY the facts explicitly stated in the Context below.
   - If the Context does NOT contain information to answer the question (e.g., burns, injuries, or any condition/topic not found in the Context), you MUST state:
     "I could not find information regarding this topic in the provided medical reference documents. Please consult a healthcare professional for guidance."
   - Under NO circumstances should you use your own pre-trained memory to supply medical advice, first aid steps, or facts that are missing from the Context.

2. Understand User Intent:
   - DIRECT / FACTUAL QUESTIONS (e.g., "What are the symptoms of [Disease]?", "What causes [Condition]?", "How to prevent [Illness]?"):
     -> Answer the question directly and specifically for that disease.
     -> DO NOT list other unrelated diseases or create an unsolicited "Possible Conditions" section. Provide only the requested facts cleanly.
   
   - SYMPTOM CHECKING / DIAGNOSTIC QUERIES (e.g., "I have fever and rash on my body", "What illness has headache and chills?"):
     -> The user is asking about an unknown condition based on their symptoms.
     -> If the symptoms match multiple diseases in the Context, list all matching conditions as potential differential possibilities.
     -> For each matching condition, highlight distinguishing features (such as rash location, onset, or key signs) according to the reference context.
     -> NEVER attribute prevention, cure, or symptoms of one disease to another.

3. Handling Typos & Spelling: Automatically infer intended medical terms if there are typos (e.g., 'haedache', 'loos of apetite', 'feverr').

4. Professional Tone & Safety: Format clearly with bullet points. Include a brief medical reminder when appropriate.

Context:
{context}

User Question:
{question}

Helpful Medical Answer:"""


def set_custom_prompt():
    """Prompt template for QA retrieval"""
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )


def load_llm():
    """Loads the open-source Hugging Face transformer via serverless inference endpoint"""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    
    if not hf_token or not hf_token.strip():
        raise ValueError(
            "Hugging Face API Token not found!\n"
            "Please create a free token at https://huggingface.co/settings/tokens\n"
            "and add it to your .env file:\n"
            "HUGGINGFACEHUB_API_TOKEN=hf_your_token_here"
        )

    endpoint = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=hf_token.strip(),
        temperature=0.2,
        max_new_tokens=512,
        timeout=120,
    )
    return ChatHuggingFace(llm=endpoint)


def build_hybrid_retriever(db):
    """Builds an Ensemble Retriever combining stemmed keyword matching (BM25) with semantic vector search (FAISS)"""
    all_docs = list(db.docstore._dict.values())
    bm25 = BM25Retriever.from_documents(all_docs, preprocess_func=bm25_preprocess)
    bm25.k = 5

    faiss_ret = db.as_retriever(search_kwargs={'k': 5})
    return EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.5, 0.5])


def retrieval_qa_chain(llm, prompt, db):
    """Builds the RetrievalQA chain with Hybrid Search (BM25 + FAISS)"""
    retriever = build_hybrid_retriever(db)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )


def qa_bot():
    """Initializes embeddings, vector store, and the QA chain"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)


# Global chain cache to ensure fast responses
GLOBAL_CHAIN = None


def get_qa_chain():
    global GLOBAL_CHAIN
    if GLOBAL_CHAIN is None:
        GLOBAL_CHAIN = qa_bot()
    return GLOBAL_CHAIN


# Quick starter prompts for medical queries supported by the dataset
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Dengue Symptoms",
            message="What are the symptoms and signs of Dengue Fever?",
        ),
        cl.Starter(
            label="Viral Fever Symptoms",
            message="What are the symptoms of viral fever?",
        ),
        cl.Starter(
            label="Malaria Prevention",
            message="How can Malaria be prevented and controlled?",
        ),
        cl.Starter(
            label="Typhoid Causes",
            message="What causes Typhoid and what are its symptoms?",
        ),
    ]


@cl.on_chat_start
async def start():
    """Session start: caches chain without sending an unprompted message to preserve clean UI"""
    try:
        chain = get_qa_chain()
        cl.user_session.set("chain", chain)
    except Exception as e:
        # If token is missing, inform the user with friendly instructions
        cl.user_session.set("init_error", str(e))


@cl.on_message
async def main(message: cl.Message):
    """Handles incoming user queries"""
    init_error = cl.user_session.get("init_error")
    if init_error:
        await cl.Message(
            content=(
                f"⚠️ **Setup Required:**\n\n{init_error}\n\n"
                "After adding your free token to the `.env` file, please restart the server."
            )
        ).send()
        return

    chain = cl.user_session.get("chain")
    if chain is None:
        try:
            chain = get_qa_chain()
            cl.user_session.set("chain", chain)
        except Exception as e:
            await cl.Message(content=f"⚠️ **Error initializing model:** {str(e)}").send()
            return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    try:
        res = await chain.ainvoke({"query": message.content}, config={"callbacks": [cb]})
        answer = res["result"]
        sources = res.get("source_documents", [])

        # Check if the response indicates the topic wasn't found in the database
        not_found_phrases = [
            "could not find information",
            "do not contain information",
            "does not contain information",
            "not found in the provided",
            "not available in the provided",
            "no information regarding",
        ]
        is_not_found = any(phrase in answer.lower() for phrase in not_found_phrases)

        # Format sources cleanly (resource name only) only when information is actually found in documents
        if sources and not is_not_found:
            source_text = "\n\n📚 **Sources:**\n"
            seen_sources = set()
            for doc in sources:
                src = doc.metadata.get("source", "Medical Reference")
                citation = f"- {src}"
                if citation not in seen_sources:
                    seen_sources.add(citation)
                    source_text += citation + "\n"
            answer += source_text

        await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
