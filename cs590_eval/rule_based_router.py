"""Rule-based router for task classification - matches router_rules/router.py logic."""
import re
from typing import Dict, Any

# Keywords for each task type (same as router_rules/router.py)
REASONING_KEYWORDS = [
    # Math/calculation
    'calculate', 'compute', 'solve', 'find', 'determine', 'prove',
    'derive', 'evaluate', 'simplify', 'equation', 'formula',
    # Logic
    'logic', 'reasoning', 'infer', 'deduce', 'conclude', 'therefore',
    # Problem-solving
    'problem', 'solution', 'answer', 'result', 'step', 'method',
    # Math symbols/patterns
    r'\d+\s*[\+\-\*\/\=]\s*\d+',  # regex for math operations
    r'\b\d+\s*%',  # percentages
    'sqrt', 'log', 'sin', 'cos', 'tan',
    # Science
    'physics', 'chemistry', 'biology', 'science', 'experiment',
    # Arc-C specific
    'most likely', 'best explains', 'why', 'because', 'reason',
    'cause', 'effect', 'relationship'
]

FACTUAL_QA_KEYWORDS = [
    # Question words
    'who', 'what', 'when', 'where', 'which', 'whose',
    # Trivia/facts
    'name', 'called', 'known as', 'famous', 'known for',
    'located', 'capital', 'country', 'city', 'place',
    'year', 'date', 'century', 'era', 'period', 'time',
    'author', 'wrote', 'written', 'book', 'novel',
    'invented', 'discovered', 'founded', 'created',
    'president', 'king', 'queen', 'leader', 'ruler',
    # Trivia indicators
    'according to', 'in the', 'of the', 'was the', 'were the',
    'history', 'historical', 'ancient', 'modern',
]

INSTRUCTION_FOLLOWING_KEYWORDS = [
    # Action verbs
    'write', 'create', 'generate', 'make', 'produce', 'compose',
    'list', 'describe', 'explain', 'summarize', 'outline',
    'format', 'rewrite', 'rephrase', 'translate', 'convert',
    'give', 'provide', 'show', 'demonstrate',
    # Formatting/structure
    'format', 'style', 'structure', 'organize', 'arrange',
    'bullet', 'numbered', 'paragraph', 'sentence', 'word',
    'capital', 'lowercase', 'uppercase', 'bold', 'italic',
    # Instructions
    'follow', 'according to', 'must', 'should', 'requirement',
    'instruction', 'constraint', 'rule', 'guideline',
    'exactly', 'precisely', 'specifically',
    # Output requests
    'output', 'response', 'reply', 'answer in',
]

# Compile regex patterns once at module load
_reasoning_patterns = [
    re.compile(kw, re.IGNORECASE) if '\\' in kw or '[' in kw
    else re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    for kw in REASONING_KEYWORDS
]
_factual_qa_patterns = [
    re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    for kw in FACTUAL_QA_KEYWORDS
]
_instruction_patterns = [
    re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    for kw in INSTRUCTION_FOLLOWING_KEYWORDS
]


def route_question(item: Dict[str, Any]) -> str:
    """
    Route a question to task type using keyword-based scoring.

    Uses same logic as router_rules/router.py for consistency.

    Args:
        item: Dict with 'question' or 'prompt' field

    Returns:
        'reasoning', 'factual_qa', or 'instruction_following'
    """
    # Get question text
    question = str(item.get('question') or item.get('prompt', '')).strip()

    if not question:
        return 'factual_qa'  # Default

    # Count keyword matches for each category
    reasoning_score = sum(1 for p in _reasoning_patterns if p.search(question))
    factual_qa_score = sum(1 for p in _factual_qa_patterns if p.search(question))
    instruction_score = sum(1 for p in _instruction_patterns if p.search(question))

    # Heuristic: Check question patterns
    # Questions starting with "What/Who/When/Where/Which" are likely factual
    if re.match(r'^\s*(what|who|when|where|which)\b', question, re.IGNORECASE):
        factual_qa_score += 2

    # Questions with "how to" or imperative verbs are likely instructions
    if re.search(r'\b(how to|please|write|create|generate)\b', question, re.IGNORECASE):
        instruction_score += 2

    # Math/calculation patterns are reasoning
    if re.search(r'\d+\s*[\+\-\*\/\=]|\d+%|calculate|solve', question, re.IGNORECASE):
        reasoning_score += 3

    # Multiple choice patterns with "Question" and options A) B) C) D) strongly suggest reasoning (ARC-C)
    has_question_word = bool(re.search(r'\bquestions?\b', question, re.IGNORECASE))
    option_count = len(re.findall(r'\b[A-D]\)', question))

    if has_question_word and option_count >= 2:
        # Strong indicator of ARC-C style multiple choice
        reasoning_score += 5
    elif option_count >= 3:
        # Multiple options without "Question" word still likely reasoning
        reasoning_score += 3
    elif option_count >= 2:
        reasoning_score += 2

    # Return the category with highest score
    scores = {
        'reasoning': reasoning_score,
        'factual_qa': factual_qa_score,
        'instruction_following': instruction_score
    }

    # Default to factual_qa if all scores are 0
    if max(scores.values()) == 0:
        return 'factual_qa'

    return max(scores, key=scores.get)
