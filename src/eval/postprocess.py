import re, string


def extract_letter(s: str) -> str | None:
    """Returns the first occurence of a letter (option/choice)::USAGE only for ARC-C

    Args:
        s (str): The output from the model for ARC-C

    Returns:
        str | None: Returns the choice selected - capital letter from ['A', 'B', 'C', 'D']
    """
    _LETTER_RE = re.compile(r"\b([ABCD])\b", re.I)
    m = _LETTER_RE.search(s.strip())
    return m.group(1).upper() if m else None



############################CHK these functions below!!!####################

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile("[" + re.escape(string.punctuation) + "]")

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def trivia_first_span(s: str) -> str:
    # take first line / sentence fragment; keep it short
    s = s.strip().splitlines()[0]
    # stop at common end tokens
    m = re.split(r"(?<=[\.\!\?])\s|[,;]|\\n", s)[0]
    return m.strip()

def em(a: str, b: str) -> int:
    return int(normalize_text(a) == normalize_text(b))

def f1(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return float(a == b)
    common = {}
    for w in ta:
        common[w] = min(ta.count(w), tb.count(w))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def ifeval_rule_check(response: str, rule: dict) -> bool:
    # Minimal rule engine: supports must_contain, must_not_contain, max_tokens
    ok = True
    r = {k: v for k, v in (rule or {}).items()}
    if "must_contain" in r:
        ok &= all(t.lower() in response.lower() for t in r["must_contain"])
    if "must_not_contain" in r:
        ok &= all(t.lower() not in response.lower() for t in r["must_not_contain"])
    if "max_tokens" in r:
        ok &= len(response.split()) <= int(r["max_tokens"])
    return ok
