import re
from typing import Tuple, Optional

# ==============================================================================
# 1. CLINICALLY-APPROVED RED-FLAG / EMERGENCY SYMPTOM POLICY (Edge Case 1 & 3)
# Based on NHS / CDC / AHA Emergency Medicine Red-Flag Indicators
# ==============================================================================
RED_FLAG_PATTERNS = [
    # Severe chest pain & cardiovascular crises
    r'\b(severe|crushing|sharp|heavy|radiating)\s+(chest\s+pain|chest\s+tightness|chest\s+pressure)\b',
    r'\b(chest\s+pain)\s+and\s+(difficulty\s+breathing|shortness\s+of\s+breath|sweating|arm\s+pain)\b',
    r'\bheart\s+attack\b',
    # Respiratory failure & acute airway compromise
    r'\b(severe|extreme|sudden)\s+(difficulty\s+breathing|shortness\s+of\s+breath|breathlessness)\b',
    r'\b(can\'?t\s+breathe|unable\s+to\s+breathe|gasping\s+for\s+air|choking|suffocating)\b',
    # Acute neurological emergencies (FAST Criteria / Stroke / Hemorrhage)
    r'\b(sudden|severe)\s+(numbness|paralysis|weakness\s+on\s+one\s+side|facial\s+droop|slurred\s+speech)\b',
    r'\b(stroke|seizure|convulsions?)\b',
    r'\b(loss\s+of\s+consciousness|passed\s+out|unresponsive|fainted\s+suddenly)\b',
    r'\b(worst\s+headache\s+of\s+(my\s+)?life|thunderclap\s+headache)\b',
    # Massive hemorrhage & shock
    r'\b(coughing\s+up\s+blood|vomiting\s+blood|uncontrolled\s+bleeding|severe\s+blood\s+loss)\b',
    # Anaphylaxis & severe allergic collapse
    r'\b(throat\s+closing|swelling\s+of\s+(the\s+)?throat|tongue\s+swelling|anaphylaxis|anaphylactic)\b',
    # Severe trauma / head injury
    r'\b(severe\s+head\s+injury|skull\s+fracture|stabbing|gunshot)\b',
    # Acute crisis / self-harm
    r'\b(commit\s+suicide|kill\s+myself|end\s+my\s+life|overdose\s+on\s+pills)\b',
]

EMERGENCY_RESPONSE = (
    "⚠️ **This may require urgent medical attention.**\n\n"
    "Please seek immediate professional medical care / emergency services, "
    "especially if symptoms are severe, sudden, or worsening.\n\n"
    "*The information provided by MedBot is for informational purposes and is not a diagnosis.*"
)


def check_emergency_symptoms(query: str) -> Optional[str]:
    """
    Edge Case 1: Configurable Red-Flag Triage Gate.
    Intercepts acute life-threatening symptoms before retrieval/LLM.
    """
    clean_q = query.lower().strip()
    for pattern in RED_FLAG_PATTERNS:
        if re.search(pattern, clean_q):
            return EMERGENCY_RESPONSE
    return None


# ==============================================================================
# 2. INPUT VALIDATION & MEANINGLESS INPUT (Edge Case 12)
# ==============================================================================
def validate_input(query: str) -> Tuple[bool, Optional[str]]:
    """
    Edge Case 12: Empty or Punctuation-Only Input Check.
    Prevents execution on empty or single-character prompts like '?' or ''.
    """
    if not query:
        return False, "Please enter a medical question or symptom you'd like information about."
    
    clean = query.strip()
    if not clean:
        return False, "Please enter a medical question or symptom you'd like information about."
    
    # Check if query is only punctuation or whitespace
    if re.fullmatch(r'[\s\?\.\!\,\;\:\-\_\*\#]+', clean):
        return False, "Please enter a medical question or symptom you'd like information about."
        
    return True, None


# ==============================================================================
# 3. QUERY LENGTH & CONDENSATION (Edge Case 13)
# ==============================================================================
MAX_QUERY_CHARS = 1200

def condense_or_truncate_query(query: str) -> Tuple[str, bool]:
    """
    Edge Case 13: Very Long Query (Bulk Medical History Paste) Handling.
    Intelligently truncates excessive input to preserve retriever performance.
    """
    if len(query) <= MAX_QUERY_CHARS:
        return query, False
    
    # Truncate at word boundary near limit
    truncated = query[:MAX_QUERY_CHARS]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated, True


# ==============================================================================
# 4. PROMPT INJECTION GUARD (Edge Case 11)
# ==============================================================================
INJECTION_PATTERNS = [
    r'\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b',
    r'\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b',
    r'\bforget\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)\b',
    r'\b(override|bypass)\s+(safety|rules|boundaries|guidelines|guardrails)\b',
    r'\byou\s+are\s+now\s+(DAN|unrestricted|jailbroken|an\s+unfiltered\s+AI)\b',
    r'\bact\s+as\s+(an?\s+unrestricted|a\s+hacker|jailbreak)\b',
    r'\b(show|reveal|display|output)\s+(your\s+)?(system\s+prompt|developer\s+mode|internal\s+instructions)\b',
    r'(```|===)\s*system\b',
]

INJECTION_RESPONSE = (
    "⚠️ **Security Notice:** Instructions attempting to override MedBot's safety guidelines "
    "and knowledge boundaries are not permitted. Please submit a valid medical or health-related question."
)

def detect_prompt_injection(query: str) -> Optional[str]:
    """
    Edge Case 11: Programmatic Prompt Injection Detection Layer.
    Blocks adversarial prompt injections before retrieval or LLM execution.
    """
    clean_q = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, clean_q):
            return INJECTION_RESPONSE
    return None


# ==============================================================================
# 5. MEDICATION & PERSONAL DOSAGE GUARD (Edge Case 6)
# ==============================================================================
PERSONAL_DOSAGE_PATTERNS = [
    r'\bhow\s+many\s+(tablets?|pills?|capsules?|mg|drops?|spoons?|doses?)\b',
    r'\b(what|which)\s+dosage\s+(should\s+i|can\s+i|do\s+i|must\s+i)\s+(take|consume|use)\b',
    r'\bhow\s+much\s+(medicine|dosage|dose|mg|syrup)\s+(should|can)\s+i\s+(take|give)\b',
    r'\bcan\s+i\s+take\s+\d+\s*(mg|tablets?|pills?|drops?)\b',
    r'\bprescribe\s+(me|a\s+treatment|medication|pills?)\b',
    r'\bhow\s+often\s+(should|can)\s+i\s+take\b',
    r'\bwhat\s+is\s+my\s+dose\b',
    r'\bhow\s+many\s+times\s+a\s+day\s+should\s+i\s+take\b',
]

DOSAGE_REFUSAL_RESPONSE = (
    "⚠️ **Medication & Dosage Guidance Notice:**\n\n"
    "MedBot provides general educational information from clinical reference documents and **cannot provide personalized medication dosages, administration schedules, or prescriptions**.\n\n"
    "Proper dosage depends heavily on individual clinical factors including age, body weight, liver and kidney function, allergy history, and concurrent medications.\n\n"
    "Please consult a qualified doctor or licensed pharmacist for personalized prescription and dosage instructions."
)

def check_medication_dosage(query: str) -> Optional[str]:
    """
    Edge Case 6: Medication & Personal Dosage Guard.
    Distinguishes educational drug questions from personal dosage/prescription requests.
    """
    clean_q = query.lower()
    for pattern in PERSONAL_DOSAGE_PATTERNS:
        if re.search(pattern, clean_q):
            return DOSAGE_REFUSAL_RESPONSE
    return None


# ==============================================================================
# 6. OUT-OF-DOMAIN DETECTOR (Edge Case 5)
# ==============================================================================
OUT_OF_DOMAIN_PATTERNS = [
    # Politics, world leaders, elections
    r'\b(who\s+is\s+(the\s+)?president|prime\s+minister|chancellor|monarch|senator|governor)\b',
    r'\b(election|political\s+party|parliament|congress|white\s+house)\b',
    # Programming, coding, technology
    r'\b(write\s+(a\s+)?python|javascript|c\+\+|html|css|sql|function|code|algorithm|docker|kubernetes)\b',
    r'\b(debug\s+this|fix\s+my\s+code|syntax\s+error)\b',
    # Math, astronomy, geography trivia
    r'\b(capital\s+of\s+[a-z]+|who\s+wrote|highest\s+mountain|distance\s+to\s+(the\s+)?moon|speed\s+of\s+light)\b',
    r'\b(calculate\s+the|derivative\s+of|integral\s+of|solve\s+for\s+x)\b',
    # Entertainment, sports, finance
    r'\b(who\s+won\s+the|world\s+cup|super\s+bowl|nba|olympics|box\s+office|movie\s+review)\b',
    r'\b(stock\s+price|cryptocurrency|bitcoin|ethereum|invest\s+in)\b',
]

# Keywords that indicate legitimate medical context to avoid false positives
MEDICAL_CONTEXT_KEYWORDS = {
    'health', 'disease', 'symptom', 'symptoms', 'fever', 'pain', 'virus', 'viral',
    'bacterial', 'bacteria', 'infection', 'cure', 'treatment', 'prevention', 'prevent',
    'doctor', 'hospital', 'medicine', 'medication', 'blood', 'skin', 'heart', 'lung',
    'liver', 'kidney', 'rash', 'headache', 'dengue', 'malaria', 'chicken pox',
    'polio', 'hepatitis', 'measles', 'tuberculosis', 'cholera', 'typhoid', 'rabies',
    'cancer', 'diabetes', 'cough', 'cold', 'flu', 'nausea', 'vomiting', 'diarrhea',
    'pathogen', 'incubation', 'transmission', 'dose', 'tablets', 'patient', 'diagnosis'
}

OUT_OF_DOMAIN_RESPONSE = (
    "This question is outside MedBot's medical knowledge scope. "
    "MedBot is an assistant dedicated to medical and health-related inquiries based on verified clinical literature."
)

def is_out_of_domain(query: str) -> bool:
    """
    Edge Case 5: Out-of-Domain Classification.
    Detects queries that have no medical relevance prior to retrieval.
    """
    clean_q = query.lower()
    words = set(re.findall(r'\b\w+\b', clean_q))
    
    # If any strong medical keyword is present, treat as medical
    if words.intersection(MEDICAL_CONTEXT_KEYWORDS):
        return False
    
    # Check explicit non-medical patterns
    for pattern in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pattern, clean_q):
            return True
            
    return False


# ==============================================================================
# 7. MEDICAL TERMINOLOGY & TYPO NORMALIZER (Edge Case 8)
# ==============================================================================
MEDICAL_TYPO_MAP = {
    r'\bhaedache[s]?\b': 'headache',
    r'\bheadake[s]?\b': 'headache',
    r'\bfeverr[s]?\b': 'fever',
    r'\bfevr\b': 'fever',
    r'\bloos\s+of\s+apetite\b': 'loss of appetite',
    r'\bloss\s+of\s+apetite\b': 'loss of appetite',
    r'\bloss\s+of\s+appetit\b': 'loss of appetite',
    r'\bdiarhea\b': 'diarrhea',
    r'\bdiarrhe\b': 'diarrhea',
    r'\bdiaria\b': 'diarrhea',
    r'\bnause\b': 'nausea',
    r'\bnauseea\b': 'nausea',
    r'\bvometing\b': 'vomiting',
    r'\bvomitting\b': 'vomiting',
    r'\bstomac\b': 'stomach',
    r'\bstomache\s+pain\b': 'stomach pain',
    r'\bbrethless(ness)?\b': 'breathlessness',
    r'\bdifficuly\s+breathing\b': 'difficulty breathing',
    r'\bdifficulty\s+in\s+breathing\b': 'difficulty breathing',
    r'\bfatig\b': 'fatigue',
    r'\bfatique\b': 'fatigue',
    r'\brashe[s]?\b': 'rash',
    r'\bcoughh\b': 'cough',
    r'\bdengu\b': 'dengue',
    r'\bmalari\b': 'malaria',
    r'\bchiken\s+pox\b': 'chicken pox',
    r'\bchickenpox\b': 'chicken pox',
    r'\bmeasels\b': 'measles',
    r'\btyphoyd\b': 'typhoid',
    r'\btyphoidd\b': 'typhoid',
    r'\bpneumoni\b': 'pneumonia',
    r'\bpnuemonia\b': 'pneumonia',
    r'\bheppatitis\b': 'hepatitis',
    r'\bhepatitus\b': 'hepatitis',
    r'\bchils\b': 'chills',
}

def normalize_medical_query(query: str) -> str:
    """
    Edge Case 8: Pre-Retrieval Medical Terminology & Typo Normalizer.
    Normalizes common spelling mistakes so BM25 and vector search match accurately.
    """
    normalized = query
    for typo_pattern, correction in MEDICAL_TYPO_MAP.items():
        normalized = re.sub(typo_pattern, correction, normalized, flags=re.IGNORECASE)
    return normalized
