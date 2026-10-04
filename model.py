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

# Import comprehensive safety, triage, and normalization guardrails
from safety import (
    check_emergency_symptoms,
    validate_input,
    condense_or_truncate_query,
    detect_prompt_injection,
    check_medication_dosage,
    is_out_of_domain,
    OUT_OF_DOMAIN_RESPONSE,
    normalize_medical_query
)

# Path to the FAISS vector database
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Pretrained Open Source Transformer on Hugging Face
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"


# Morphological tokenizer for BM25: handles plurals, hyphenation, and clinical variations
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


# Clinical prompt template satisfying:
# - Edge Case 2: Normal Informational Questions
# - Edge Case 3 & 7: Symptom Inquiries & Diagnostic Limitation ("MedBot cannot confirm a diagnosis")
# - Edge Case 4: Unknown Disease / Missing Knowledge Refusal
# - Edge Case 10: Multi-source Conflicting Information Surfacing
custom_prompt_template = """You are an expert AI Medical Assistant. Your task is to provide accurate, concise, and safe medical information based strictly and exclusively on the provided reference context.

CRITICAL CLINICAL INSTRUCTIONS & SAFETY BOUNDARIES:
1. STRICT INFORMATION BOUNDARY (NO EXTERNAL KNOWLEDGE / NO HALLUCINATION):
   - You MUST answer questions using ONLY the facts explicitly stated in the Context below.
   - If the Context does NOT contain information to answer the question (e.g., burns, injuries, or any condition/topic not found in the Context), you MUST state:
     "I could not find information regarding this topic in the provided medical reference documents. Please consult a healthcare professional for guidance."
   - Under NO circumstances should you use your own pre-trained memory to supply medical advice, first aid steps, or facts that are missing from the Context.

2. DIAGNOSTIC BOUNDARIES & SYMPTOM INQUIRIES:
   - Under NO circumstances should you state "You have [Condition]" or provide a definitive clinical diagnosis.
   - For ANY symptom inquiry or diagnostic query (e.g., "I have fever and headache", "What disease do I have?"):
     -> You MUST start your response with: "These symptoms may be associated with several conditions described in the reference material. MedBot cannot confirm a diagnosis."
     -> Present matching conditions from the Context as potential differential possibilities.
     -> For each matching condition, highlight distinguishing features (such as rash location, onset, fever pattern, or specific signs) documented in the reference text.
     -> NEVER attribute prevention, cure, or symptoms of one disease to another.

3. DIRECT FACTUAL QUESTIONS:
   - For specific factual questions (e.g., "What are the symptoms of dengue?", "What is the incubation period of polio?", "How is malaria transmitted?"):
     -> Answer directly, concisely, and specifically for that condition.
     -> Do NOT issue false emergency alerts for standard educational questions.
     -> Do NOT create an unsolicited differential diagnosis section when a single specific disease was queried.

4. MEDICATION & TREATMENT SAFETY:
   - Do NOT provide personalized drug dosages, prescriptions, or individualized administration schedules.
   - General pharmacological facts from the Context must be accompanied by advice to consult a licensed physician or pharmacist.

5. CONFLICTING REFERENCE INFORMATION:
   - If multiple reference documents or sections contain contradictory, differing, or varying guidance, explicitly highlight both perspectives and cite the respective sources, rather than silently picking one.

6. FORMAT & PROFESSIONALISM:
   - Use clean markdown bullet points. Ensure high readability and clinical neutrality.

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
    ensemble = EnsembleRetriever(retrievers=[bm25, faiss_ret], weights=[0.5, 0.5])
    return ensemble, bm25


def evaluate_retrieval_confidence(query: str, db, bm25, max_l2_dist: float = 1.15):
    """
    Edge Case 9: Retrieval Confidence Gate.
    Verifies that the retrieved documents meet relevance thresholds before passing to LLM.
    If the context is irrelevant or distance is excessive, returns False to trigger grounded refusal.
    """
    faiss_res = db.similarity_search_with_score(query, k=4)
    if not faiss_res:
        return False, 999.0, []
    
    best_distance = faiss_res[0][1]
    retrieved_docs = [doc for doc, _ in faiss_res]
    
    # Highly confident dense semantic match
    if best_distance <= 0.95:
        return True, best_distance, retrieved_docs
        
    # Moderate vector similarity: check if keyword match exists in BM25
    q_words = set(re.findall(r'\w+', query.lower())) - {
        'what', 'are', 'the', 'is', 'of', 'in', 'and', 'to', 'for', 'about', 
        'tell', 'me', 'i', 'have', 'does', 'do', 'can', 'with', 'from', 'my'
    }
    meaningful_words = [w for w in q_words if len(w) > 3]
    
    bm25_docs = bm25.invoke(query)
    has_keyword_match = False
    if bm25_docs and meaningful_words:
        top_context = " ".join([d.page_content for d in bm25_docs[:2]]).lower()
        if any(w in top_context for w in meaningful_words):
            has_keyword_match = True
            
    if best_distance <= max_l2_dist and has_keyword_match:
        return True, best_distance, retrieved_docs
        
    # Below confidence threshold
    return False, best_distance, retrieved_docs


def retrieval_qa_chain(llm, prompt, db):
    """Builds the RetrievalQA chain with Hybrid Search (BM25 + FAISS)"""
    ensemble, bm25 = build_hybrid_retriever(db)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=ensemble,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return chain, bm25


def qa_bot():
    """Initializes embeddings, vector store, and the QA chain components"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    chain, bm25 = retrieval_qa_chain(llm, qa_prompt, db)
    return (chain, db, bm25)


# Global chain cache to ensure fast responses
GLOBAL_CHAIN_COMPONENTS = None


def get_qa_chain():
    global GLOBAL_CHAIN_COMPONENTS
    if GLOBAL_CHAIN_COMPONENTS is None:
        GLOBAL_CHAIN_COMPONENTS = qa_bot()
    return GLOBAL_CHAIN_COMPONENTS


@cl.on_chat_start
async def start():
    """Session start: caches chain without sending an unprompted message to preserve clean UI"""
    try:
        chain_components = get_qa_chain()
        cl.user_session.set("chain_components", chain_components)
    except Exception as e:
        # Edge Case 14: Graceful initialization error handling
        cl.user_session.set("init_error", str(e))


@cl.on_message
async def main(message: cl.Message):
    """
    Handles incoming user queries through a multi-stage clinical guardrail pipeline:
    1. Input Validation (Edge Case 12)
    2. Query Condensation & Truncation (Edge Case 13)
    3. Prompt Injection Detection (Edge Case 11)
    4. Emergency / Red-Flag Triage Gate (Edge Case 1 & 3)
    5. Medication & Dosage Guard (Edge Case 6)
    6. Out-of-Domain Classification (Edge Case 5)
    7. Typo & Medical Terminology Normalization (Edge Case 8)
    8. Retrieval Confidence Gate (Edge Case 9)
    9. Grounded LLM Generation (Edge Cases 2, 3, 4, 7, 10)
    10. Granular Source Transparency (Edge Case 15)
    11. Graceful API Failure Handling (Edge Case 14)
    """
    user_query = message.content or ""

    # Stage 1: Edge Case 12 - Empty or Punctuation-Only Input Check
    is_valid, validation_msg = validate_input(user_query)
    if not is_valid:
        await cl.Message(content=validation_msg).send()
        return

    # Stage 2: Edge Case 13 - Very Long Query Handling (Bulk Medical History Paste)
    condensed_query, was_truncated = condense_or_truncate_query(user_query)
    truncation_notice = "*(Note: Extensive query detected. MedBot processed the primary symptoms against reference documentation.)*\n\n" if was_truncated else ""

    # Stage 3: Edge Case 11 - Prompt Injection Detection Layer
    injection_resp = detect_prompt_injection(condensed_query)
    if injection_resp:
        await cl.Message(content=injection_resp).send()
        return

    # Stage 4: Edge Case 1 & 3 - Clinician-Approved Red-Flag / Emergency Symptom Triage Gate
    emergency_resp = check_emergency_symptoms(condensed_query)
    if emergency_resp:
        await cl.Message(content=emergency_resp).send()
        return

    # Stage 5: Edge Case 6 - Medication & Personal Dosage Guard
    dosage_resp = check_medication_dosage(condensed_query)
    if dosage_resp:
        await cl.Message(content=dosage_resp).send()
        return

    # Stage 6: Edge Case 5 - Out-of-Domain Classifier
    if is_out_of_domain(condensed_query):
        await cl.Message(content=OUT_OF_DOMAIN_RESPONSE).send()
        return

    # Stage 7: Edge Case 8 - Pre-Retrieval Medical Terminology & Typo Normalization
    normalized_query = normalize_medical_query(condensed_query)

    # Check setup / initialization errors
    init_error = cl.user_session.get("init_error")
    if init_error:
        await cl.Message(
            content=(
                f"⚠️ **Setup Required:**\n\n{init_error}\n\n"
                "After adding your token to the `.env` file, please restart the server."
            )
        ).send()
        return

    chain_components = cl.user_session.get("chain_components")
    if chain_components is None:
        try:
            chain_components = get_qa_chain()
            cl.user_session.set("chain_components", chain_components)
        except Exception:
            # Edge Case 14: Graceful error without exposing raw traceback
            await cl.Message(
                content="⚠️ **Service temporarily unavailable.**\n\nModel initialization failed. Please verify system credentials and restart the service."
            ).send()
            return

    chain, db, bm25 = chain_components

    # Stage 8: Edge Case 9 - Retrieval Confidence Gate
    is_confident, best_distance, _ = evaluate_retrieval_confidence(normalized_query, db, bm25)
    if not is_confident:
        await cl.Message(
            content="I could not find information regarding this topic in the provided medical reference documents. Please consult a healthcare professional for guidance."
        ).send()
        return

    # Stage 9: Execute Generation with Token Streaming
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    try:
        res = await chain.ainvoke({"query": normalized_query}, config={"callbacks": [cb]})
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

        # Stage 10: Edge Case 15 - Granular Source Transparency (Page-level & Document-level Citations)
        if sources and not is_not_found:
            source_lines = []
            seen_citations = set()
            for doc in sources:
                doc_name = doc.metadata.get("document_name") or os.path.basename(doc.metadata.get("source", "Medical Reference"))
                page_num = doc.metadata.get("page")
                if page_num:
                    citation = f"- **{doc_name}** (Page {page_num})"
                else:
                    citation = f"- **{doc_name}**"
                if citation not in seen_citations:
                    seen_citations.add(citation)
                    source_lines.append(citation)
            if source_lines:
                answer += "\n\n📚 **Sources:**\n" + "\n".join(source_lines)

        final_content = truncation_notice + answer
        await cl.Message(content=final_content).send()

    except Exception as e:
        # Stage 11: Edge Case 14 - Graceful API & Runtime Error Handling
        print(f"[MedBot System Error]: {e}")
        await cl.Message(
            content="⚠️ **Service temporarily unavailable.**\n\nPlease check your network connection or try again in a few moments."
        ).send()
