# normalize_output2.py
import re
import ast
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Callable

from dateutil import parser
from word2number import w2n

DNA = "data not available"

FIELDS = [
    "ConvCourtName",
    "ConvictPleaDate",
    "ConvictOffence",
    "AcquitOffence",
    "ConfessPleadGuilty",
    "PleaPoint",
    "RemandDecision",
    "RemandCustodyTime",
    "SentCourtName",
    "Sentence",
    "SentServe",
    "WhatAncillary",
    "OffSex",
    "OffAgeOffence",
    "OffJobOffence",
    "OffHomeOffence",
    "OffMentalOffence",
    "OffIntoxOffence",
    "OffVicRelation",
    "VictimType",
    "VicNum",
    "VicSex",
    "VicAgeOffence",
    "VicJobOffence",
    "VicHomeOffence",
    "VicMentalOffence",
    "VicIntoxOffence",
    "ProsEvidTypeTrial",
    "DefEvidTypeTrial",
    "PreSentReport",
    "AggFactSent",
    "MitFactSent",
    "VicImpactStatement",
    "Appellant",
    "CoDefAccNum",
    "AppealAgainst",
    "AppealGround",
    "SentGuideWhich",
    "AppealOutcome",
    "ReasonQuashConv",
    "ReasonSentExcessNotLenient",
    "ReasonSentLenientNotExcess",
    "ReasonDismiss",
]

# -----------------------------
# Helpers
# -----------------------------
def _is_dna(x: Any) -> bool:
    return isinstance(x, str) and x.strip().lower() == DNA

def _is_dk(x: Any) -> bool:
    return isinstance(x, str) and x.strip().lower() in {"don't know", "dont know"}

def _clean_str(x: str) -> str:
    s = x.strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out

def _ensure_list(x: Any) -> List[str]:
    """
    Always return a list of strings.
    - None/empty -> [DNA]
    - list -> list[str] (or [DNA] if empty)
    - string -> if looks like list repr, parse; else wrap
    """
    if x is None:
        return [DNA]
    if isinstance(x, list):
        return [str(v) for v in x] if x else [DNA]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return [DNA]
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed] if parsed else [DNA]
            except Exception:
                pass
        return [x]
    return [str(x)]

def _normalize_sentinel_value(field: str, v: str) -> str:
    """
    Keep sentinel values stable/canonical for downstream scoring.
    """
    if _is_dna(v):
        return DNA
    if _is_dk(v):
        # your conventions: OffSex/VicSex use "Don't Know" (capital K)
        if field in {"OffSex", "VicSex"}:
            return "Don't Know"
        return "Don't know"
    return _clean_str(v)

# -----------------------------
# Converters (your normalization logic)
# -----------------------------
def convert_date(x: Any) -> Any:
    if not isinstance(x, str):
        return None

    s = x.strip()
    if not s:
        return None

    low = s.lower()
    if low in ("don't know", "dont know", "data not available"):
        return s

    s = s.replace("\xa0", " ")
    s = re.sub(r"[\r\n]+", " ", s)
    s = s.rstrip(",").strip()
    s = re.sub(r"^\s*on\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", s, flags=re.IGNORECASE)

    default_dt = datetime(1900, 1, 1)

    # Between 2001 and 2013 → [2001-01-01, 2013-01-01]
    # m_years = re.match(r"^(?i)between\s+(\d{4})\s+and\s+(\d{4})$", s)
    m_years = re.match(r"^between\s+(\d{4})\s+and\s+(\d{4})$", s, flags=re.IGNORECASE)

    if m_years:
        y1, y2 = m_years.groups()
        return [f"{y1}-01-01", f"{y2}-01-01"]

    # DD.MM.YYYY
    m_dmy = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{2,4})$", s)
    if m_dmy:
        day, month, yr = map(int, m_dmy.groups())
        try:
            return date(yr, month, day).isoformat()
        except ValueError:
            return None

    # 11 and 12 April 2012 → [2012-04-11, 2012-04-12]
    m_dom_range = re.match(r"^(\d{1,2})\s+and\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", s, flags=re.IGNORECASE)
    if m_dom_range:
        d1, d2, mon, yr = m_dom_range.groups()
        try:
            mon_dt = parser.parse(f"{mon} {yr}", default=default_dt)
            mnum = mon_dt.month
            return [
                date(int(yr), mnum, int(d1)).isoformat(),
                date(int(yr), mnum, int(d2)).isoformat(),
            ]
        except Exception:
            return None

    # Multi-month ranges: “October and November 2008” → [2008-10-01, 2008-11-01]
    year_match = re.search(r"\b(\d{4})\b", s)
    year = year_match.group(1) if year_match else None
    if " and " in s and year:
        parts = [p.strip() for p in re.split(r"\band\b", s, flags=re.IGNORECASE)]
        results = []
        for part in parts:
            if not re.search(r"\b\d{4}\b", part):
                part = f"{part} {year}"
            try:
                dt = parser.parse(part, dayfirst=True, fuzzy=True, default=default_dt)
                results.append(dt.date().isoformat())
            except Exception:
                continue
        return results or None

    # Fallback parse
    try:
        dt = parser.parse(s, dayfirst=True, fuzzy=True, default=default_dt)
        return dt.date().isoformat()
    except Exception:
        return None

def convert_confess_to_yes_no(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip()
    low = s.lower()

    if low in {"don't know", "dont know", "data not available"}:
        return s

    # NEGATIVE first
    if re.search(r"\bnot guilty\b", low) or re.search(r"\bdenied\b", low) or re.match(r"^no\b", low):
        return "No"

    # POSITIVE patterns
    if (
        re.search(r"\bconfess(?:ed|ions?)?\b", low)
        or re.search(r"\badmit(?:ted)?\b", low)
        or re.search(r"\bguilty plea(?:s)?\b", low)
        or re.search(r"\bplea\b", low)
        or re.search(r"\bplead(?:ed|s|ing)?\b", low)
        or re.search(r"\bguilty\b", low)
        or re.match(r"^yes\b", low)
    ):
        return "Yes"

    return "Don't know"

def convert_sentence_order(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    low = re.sub(r"[\r\n]+", " ", x.strip().lower())

    if low in {"don't know", "dont know"}:
        return "Don't know"
    if low == "data not available":
        return "data not available"

    if "combination" in low:
        return "Combination"
    if "concurrently" in low and "consecutively" in low:
        return "Combination"
    if "concurrently" in low:
        return "Concurrently"
    if "consecutively" in low:
        return "Consecutively"

    is_conc = "concurrent" in low
    is_seq = "consecutive" in low
    is_sng = "single" in low

    if is_conc and is_seq:
        return "Combination"
    if is_sng:
        return "Single"
    if is_conc:
        return "Concurrent"
    if is_seq:
        return "Consecutive"
    return "Don't know"

def convert_custody_status(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().lower()

    if s == "data not available":
        return "data not available"
    if "don't know" in s or "dont know" in s:
        return "Don't know"

    if s == "yes":
        return "Remanded into custody"
    if s == "no":
        return "No"

    if "unconditional bail" in s:
        return "Unconditional Bail"
    if "bail" in s:
        return "Conditional Bail" if any(k in s for k in ("conditional", "curfew", "condition")) else "Unconditional Bail"
    if "curfew" in s:
        return "Conditional Bail"
    if any(term in s for term in ("remand", "custody", "detained", "in custody")):
        return "Remanded into custody"

    return "Don't know"

def convert_gender(x: Any) -> Any:
    if not isinstance(x, str):
        return x

    s = x.strip().lower()
    s = re.sub(r"[^\w\s&]+$", "", s)

    if s in {"don't know", "dont know"}:
        return "Don't Know"
    if s == "data not available":
        return "data not available"

    male_tokens = ['male','man','men','he','him','his','boy','mlae','males','boys','both male']
    female_tokens = ['female','woman','women','she','her','girl','females','girls','both female']

    has_male = any(re.search(rf"\b{t}\b", s) for t in male_tokens)
    has_fem  = any(re.search(rf"\b{t}\b", s) for t in female_tokens)

    if (has_male and has_fem) or ("and" in s) or ("&" in s) or ("mixed" in s):
        return "Mixed"
    if has_fem:
        return "All Female"
    if has_male:
        return "All Male"
    return "Don't Know"

def convert_age(x: Any) -> Any:
    # handle single-element list
    if isinstance(x, list) and len(x) == 1:
        return convert_age(x[0])
    if not isinstance(x, str):
        return x

    s = x.strip()
    low = s.lower().replace("\xa0", " ")

    if low in {"don't know", "dont know"}:
        return "Don't know"
    if low == "data not available":
        return "data not available"

    # Half-years: “4½”, “4 ½”, “4 1/2”
    m = re.match(r"^\s*(\d{1,3})\s*(?:½|1/2)\s*$", low)
    if m:
        base = int(m.group(1))
        age = base + 0.5
        return str(age) if age <= 100 else "Don't know"

    # “3 of _”
    m = re.search(r"\b(\d{1,3})\s+of\s+_+\b", low)
    if m:
        return f"{m.group(1)} of _"

    # “life of 44 years”
    m = re.search(r"\blife of\s+(\d{1,3})\b", low)
    if m:
        age = int(m.group(1))
        return str(age) if age <= 100 else "Don't know"

    # “mid-sixties”
    m = re.search(r"\bmid[- ]([a-z]+?)(?:ies|s)?\b", low)
    if m:
        dw = m.group(1)
        try:
            decade = w2n.word_to_num(dw)
            return f"{decade}-{decade+10}"
        except Exception:
            pass

    # “age of X”
    m = re.search(r"\bage of\s+(\d+|\w+)\b", low)
    if m:
        tok = m.group(1)
        try:
            age = int(tok) if tok.isdigit() else w2n.word_to_num(tok)
            return str(age) if age <= 100 else "Don't know"
        except Exception:
            return tok

    # “X and Y of Z”
    m = re.search(r"\b(\d{1,3})\s*and\s*(\d{1,3})\s+of\s+([_\d]+)\b", low)
    if m:
        a, b, c = m.groups()
        lo, hi = sorted((int(a), int(b)))
        return f"{lo}-{hi} of {c}"

    # “20s or 30s”
    m = re.search(r"\b(\d+)s\s*or\s*(\d+)s\b", low)
    if m:
        a, b = map(int, m.groups())
        lo, hi = sorted((a, b))
        return f"{lo}-{hi}"

    # Hyphen ranges “28-42”
    m = re.search(r"\b(\d{1,3})\s*[-–]\s*(\d{1,3})\b", s)
    if m:
        a, b = map(int, m.groups())
        if a <= 100 and b <= 100:
            return f"{a}-{b}"

    # Numeric “or/and”
    m = re.search(r"\b(\d{1,3})\s*(?:and|or|&)\s*(\d{1,3})\b", s)
    if m:
        a, b = map(int, m.groups())
        lo, hi = sorted((a, b))
        if hi <= 100:
            return f"{lo}-{hi}"

    # “two of 36”
    m = re.search(r"\b(\w+)\s+of\s+(\d{1,3})\b", low)
    if m:
        a, b = m.groups()
        try:
            n = int(a) if a.isdigit() else w2n.word_to_num(a)
            if n <= 100:
                return f"{n} of {b}"
        except Exception:
            return f"{a} of {b}"

    # first integer
    nums = re.findall(r"\b(\d{1,3})\b", s)
    if nums:
        n = int(nums[0])
        return str(n) if n <= 100 else "Don't know"

    # first written number
    try:
        n = w2n.word_to_num(low)
        return str(n) if n <= 100 else "Don't know"
    except Exception:
        return "Don't know"

def convert_occupation(x: Any) -> Any:
    if not isinstance(x, str):
        return x

    s = x.strip().rstrip(".,")
    low = s.lower()

    if low in {"don't know", "dont know"}:
        return "Don't know"
    if low == "data not available":
        return "data not available"

    # Self-employed first
    if "self" in low and "employ" in low:
        return "Employed"

    if re.search(r"\bstudent(s)?\b", low) or re.search(r"\bin training\b", low):
        return "Student"
    if re.search(r"\bchild\b", low):
        return "Child"
    if re.search(r"\bretir(?:ed|ing)?\b", low):
        return "Retired"
    if re.search(r"\bunemploy(?:ed|ment)?\b", low) or re.search(r"\bnot work(?:ed|ing)?\b", low):
        return "Unemployed"

    if re.search(
        r"\b(?:employ(?:ed|ee)?|worker|manager|driver|cleaner|tuner|"
        r"restorer|scientist|doctor|optometrist|boxer|teacher|nurse|"
        r"clerk|engineer|accountant|sales)\b",
        low,
    ):
        return "Employed"

    if "other" in low:
        return "Other"
    return "Other"

def convert_address_status(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().rstrip(".,")
    low = s.lower()

    if low in {"don't know", "dont know"}:
        return "Don't Know"
    if low == "data not available":
        return "data not available"

    if "section" in low:
        return "Sectioned"
    if "homeless" in low:
        return "Homeless"
    if any(kw in low for kw in ("temporary", "hostel", "care home", "care", "hospital")):
        return "Temporary Accommodation"
    if any(
        kw in low
        for kw in (
            "fixed address",
            "fixed addres",
            "fixed",
            "shared address",
            "applicant's",
            "flat",
            "home",
            "family home",
            "matrimonial home",
            "from his home",
            "address",
            "lived",
            "living",
        )
    ):
        return "Fixed Address"
    return "Don't Know"

def convert_disability(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    low = x.strip().lower()

    if low in {"don't know", "dont know"}:
        return "Don't know"
    if low == "data not available":
        return "data not available"

    is_learning = bool(re.search(r"\blearning\b", low))
    is_developmental = bool(re.search(r"\bdevelopmental\b", low)) or ("autism" in low)

    is_mental = bool(
        re.search(
            r"\b(?:mental health|psychiatr\w*|depress\w*|anxi\w*|bipolar\w*|"
            r"schizophren\w*|ptsd|post-traumatic stress disorder|ocd|adhd|"
            r"mental disorder|addict\w*|eating disorder|personality disorder|"
            r"psychotic|disturb\w*)\b",
            low,
        )
    )

    if is_mental:
        return "Had mental health problems"
    if is_learning and is_developmental:
        return "Learning/developmental"
    if is_learning:
        return "Has learning difficulties"
    if is_developmental:
        return "Learning/developmental"
    if "other" in low:
        return "Other"
    return "Other"

def convert_substance_use(x: Any) -> Any:
    if not isinstance(x, str):
        return x

    s = x.strip()
    low = s.lower()

    if "data not available" in low:
        return "data not available"
    if "don't know" in low or "dont know" in low:
        return "Don't know"

    has_alc = bool(
        re.search(
            r"\b(drink(?:ing|s|ed)?|alcohol(?:ic)?|vodka|beer|wine|spirit(?:s)?|"
            r"lager|whisky|liquor|pint(?:s)?|drunk|drank|drunken|intoxicated)\b",
            low,
        )
    )
    has_drugs = bool(
        re.search(
            r"\b(drug(?:s)?|narcotic(?:s)?|heroin|paracetamol|citalopram|"
            r"anti-depressant(?:s)?|depressant(?:s)?|cocaine|crack|cannabis|spice|"
            r"ketamine|benzoylecgonine|ecstasy|amphet(?:amine)?|meth(?:amphetamine)?|"
            r"opioid(?:s)?|nitrous oxide|shrooms|lsd|pcp|substance(?:s)?)\b",
            low,
        )
    ) or bool(re.search(r"\btest(?:ed)? (?:positive|provided a positive result)\b", low))

    if has_alc and has_drugs:
        return "Yes-drinking&drugs"
    if has_alc:
        return "Yes-drinking"
    if has_drugs:
        return "Yes-drugs"
    return "No"

def convert_relationship(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip()
    low = s.lower()

    if low in {"don't know", "dont know"}:
        return "Don't know"
    if low == "data not available":
        return "data not available"

    if re.search(r"\bstrang(?:er|ers)?\b", low) or ("total stranger" in low) or ("stanger" in low):
        return "Stranger"
    if re.search(
        r"\b(relat(?:ed|ionship|ive)?|grand(?:-)?|granddaughter(?:s)?|grandson(?:s)?|"
        r"spous(?:e)?|partner|ex[- ]?partner)\b",
        low,
    ):
        return "Relative"
    if re.search(r"\bacquaint(?:ance|ed)?\b", low) or re.search(r"\bfriend(?:ship|s)?\b", low):
        return "Acquaintance"
    return "Don't know"

def convert_entity(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().rstrip(",").lower()
    if s in {"don't know", "dont know"}:
        return "Don't Know"
    if s == "data not available":
        return "data not available"
    if re.search(r"\b(persons|individuals)\b", s):
        return "Individuals"
    if re.search(r"\b(person|individual)\b", s):
        return "Individual person"
    if any(k in s for k in ("shop", "company", "business", "firm", "co.", "ltd")):
        return "Company"
    if any(k in s for k in ("government", "council", "authority", "ministry", "public body")):
        return "Government"
    if "organis" in s or "organisation" in s or "org " in s:
        return "Organisation"
    return "Don't Know"

def convert_risk(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().lower()

    if s in {"don't know", "dont know"}:
        return "Don't know"
    if s == "data not available":
        return "data not available"

    high = bool(re.search(r"\b(high|serious)\b", s))
    med = bool(re.search(r"\bmedium\b", s))
    low_ = bool(re.search(r"\blow\b", s))

    if not (high or med or low_):
        return "Don't know"

    rtype = "harm" if ("harm" in s or "danger" in s) else "reoffending"
    level = "High" if high else ("Medium" if med else "Low")
    return f"{level} risk of {rtype}"

def convert_victim_statement(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().rstrip(",")
    low = s.lower()
    if low in {"don't know", "dont know"}:
        return "Don't Know"
    if low == "data not available":
        return "data not available"
    return "Yes" if (low == "yes" or "statement" in low) else "No"

def convert_role(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().rstrip(",").lower()

    if s in {"don't know", "dont know"}:
        return "Don't Know"
    if s == "data not available":
        return "data not available"
    if "offender" in s:
        return "Offender"
    if "appellant" in s:
        return "Appellant"
    if "attorney general" in s or "solicitor general" in s or "solicitor-general" in s or "attorney" in s:
        return "Attorney General"
    return "Other"

def categorise_defence_statement(x: Any) -> Any:
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    if any(term in s for term in ("denies", "denied", "denial", "does not admit", "did not admit", "denying")):
        return "Offender denies offence"
    if any(term in s for term in ("alibi", "elsewhere", "other location", "not present")):
        return "Offender claims to have alibi"
    if "no clothing found" in s:
        return "No clothing found matching offender’s"
    if "no recovery of stolen goods" in s:
        return "No recovery of stolen goods from offender"
    if any(term in s for term in ("previous ruling", "previous court case", "previously acquitted", "autrefois acquit")):
        return "Previous ruling in another court case"
    if any(term in s for term in ("consistency", "consistent evidence", "consistent testimony")):
        return "Consistency of offender evidence in court and at police interview"
    if any(term in s for term in ("previous good character", "character references", "testimonials")):
        return "Offender of previous good character"
    if any(term in s for term in ("lack of dna", "no dna", "dna evidence not found")):
        return "Lack of DNA evidence to support allegations"
    if any(term in s for term in ("lack of medical", "no medical evidence", "insufficient medical")):
        return "Lack of medical evidence to support allegations"
    if any(term in s for term in ("not credible", "unreliable witness", "inconsistencies", "unreliable victim", "credibility questioned")):
        return "Victim is not credible/is unreliable"
    if any(term in s for term in ("admissibility", "inadmissible", "unreliable evidence", "abuse of process")):
        return "Question admissibility of some item of evidence"
    if any(term in s for term in ("lesser offence", "admits lesser", "guilty to lesser", "limited role", "less culpable")):
        return "Offender admits only to lesser offence"
    if any(term in s for term in ("co-accused responsible", "blames co-defendant", "main offender")):
        return "Offender states co-accused is responsible for main offence"
    if any(term in s for term in ("left scene before", "left before", "left early")):
        return "Offender argues he/she left scene before main offence committed"
    if any(term in s for term in ("bad character evidence", "character evidence against co-accused")):
        return "Offender uses bad character evidence against co-accused"
    if "cut-throat" in s:
        return "Cut-throat defence"
    if any(term in s for term in ("self-defence", "justified action", "lawful defence", "prevention of crime", "lawful excuse")):
        return "Self-defence or Justified Action"
    if any(term in s for term in ("mental health", "insanity", "diminished responsibility", "psychiatric", "psychological")):
        return "Mental Health / Insanity"
    if any(term in s for term in ("mistaken identification", "misidentified", "wrongly identified")):
        return "Mistaken Identification"
    if any(term in s for term in ("did not intend", "no intention", "unintentional", "not deliberate")):
        return "Lack of Intention"
    if any(term in s for term in ("expert report", "expert testimony", "expert witness", "experts", "expert")):
        return "Expert Evidence"
    if any(term in s for term in ("no case to answer", "no evidence offered", "submission of no case")):
        return "No Case to Answer"
    if any(term in s for term in ("innocent association", "association innocent")):
        return "Innocent Association"
    if any(term in s for term in ("provocation", "victim aggression", "victim initiated")):
        return "Victim Provocation or Aggression"
    if any(term in s for term in ("documentary evidence", "documents submitted", "witness statements", "written records")):
        return "Documentary Evidence"
    return x

def _conv_aggfactsent(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    return re.sub(r"^[•\-\*\t\s]+", "", x).strip()

def _conv_convcourt(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    x = _clean_str(x)
    if not re.search(r"Court|Assizes", x, flags=re.IGNORECASE):
        return None
    x = re.sub(r"(?i)^on appeal from\s+", "", x).strip()
    return x.title()

def _conv_sentcourt(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    x = x.replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x).strip()

    for drop in ("Appellant2", "Harrison", "On the same day", "Before the same court", "Before The Same Court"):
        x = re.sub(re.escape(drop), "", x, flags=re.IGNORECASE).strip()

    if "," in x:
        x = x.split(",", 1)[0].strip()

    x = re.sub(r"(?i)^on appeal from\s+", "", x).strip()

    if not re.search(r"\b(Court|Assizes|courrt)\b", x, flags=re.IGNORECASE):
        return None

    return x.title()

def categorise_appeal_against(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().lower()

    excessive_terms = ["excessive", "manifestly excessive", "unduly excessive", "sentence is excessive"]
    lenient_terms = ["unduly lenient", "lenient", "sentence is lenient"]
    extension_terms = ["extension of time", "out of time", "leave to appeal", "extension of time to appeal", "time"]

    excessive = any(term in s for term in excessive_terms)
    lenient = any(term in s for term in lenient_terms)
    extension = any(term in s for term in extension_terms)

    if lenient:
        return "Sentence (is unduly lenient)"
    if excessive:
        return "Sentence (is unduly excessive)"
    if extension:
        return "Other (e.g., application for extension of time to appeal)"
    return x

def convert_appeal_outcome(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip().lower()

    dismissed_terms = ["dismiss", "dismissed", "refused", "failed", "rejected", "refuse"]
    lenient_terms = ["lenient", "more lenient", "reduce", "reduced", "substitute", "sentence replaced with more lenient"]
    excessive_terms = ["more excessive", "increased", "raised", "replace", "sentence replaced by more excessive", "increase the sentence"]

    if any(term in s for term in excessive_terms):
        return "Allowed&Sentence Replaced by More Excessive Sentence"
    if any(term in s for term in lenient_terms):
        return "Allowed&Sentence Replaced by More Lenient Sentence"
    if any(term in s for term in dismissed_terms):
        return "Dismissed-Failed-Refused"
    if "substitute" in s or "substituted with" in s:
        return "Substituted with Other Sentence"
    return x

# -----------------------------
# Field -> converter mapping
# -----------------------------
FIELD_CONVERTERS: Dict[str, Callable[[Any], Any]] = {
    "ConvCourtName": _conv_convcourt,
    "ConvictPleaDate": convert_date,
    "ConfessPleadGuilty": convert_confess_to_yes_no,
    "SentServe": convert_sentence_order,
    "RemandDecision": convert_custody_status,
    "OffSex": convert_gender,
    "VicSex": convert_gender,
    "OffAgeOffence": convert_age,
    "VicAgeOffence": convert_age,
    "VicNum": convert_age,
    "CoDefAccNum": convert_age,
    "OffJobOffence": convert_occupation,
    "VicJobOffence": convert_occupation,
    "OffHomeOffence": convert_address_status,
    "VicHomeOffence": convert_address_status,
    "OffMentalOffence": convert_disability,
    "VicMentalOffence": convert_disability,
    "OffIntoxOffence": convert_substance_use,
    "VicIntoxOffence": convert_substance_use,
    "OffVicRelation": convert_relationship,
    "VictimType": convert_entity,
    "PreSentReport": convert_risk,
    "VicImpactStatement": convert_victim_statement,
    "Appellant": convert_role,
    "DefEvidTypeTrial": categorise_defence_statement,
    "AggFactSent": _conv_aggfactsent,
    "SentCourtName": _conv_sentcourt,
    "AppealAgainst": categorise_appeal_against,
    "AppealOutcome": convert_appeal_outcome,
}

# -----------------------------
# Postprocessing (your list-collapse rules)
# -----------------------------
def _collapse_sex_list(lst: List[str]) -> List[str]:
    s = set(lst)
    if s == {"All Male", "All Female"} or s == {"All Male", "Mixed"} or s == {"All Female", "Mixed"}:
        return ["Mixed"]
    return lst

def _collapse_codefaccnum_list(lst: List[str]) -> List[str]:
    max_m = None
    max_n = None
    for x in lst:
        m = re.match(r"(\d+)\s*of\s*(\d+)", x)
        if m:
            m_val = int(m.group(2))
            max_m = m_val if max_m is None else max(max_m, m_val)
        else:
            try:
                n_val = int(x)
                max_n = n_val if max_n is None else max(max_n, n_val)
            except ValueError:
                pass

    if max_m is not None:
        return [str(max_m)]
    if max_n is not None:
        return [str(max_n)]
    return lst

def _collapse_intox_list(lst: List[str]) -> List[str]:
    if any(isinstance(x, str) and x.startswith("Yes-") for x in lst):
        parts = set()
        for x in lst:
            if isinstance(x, str) and x.startswith("Yes-"):
                _, core = x.split("-", 1)
                parts.update(core.split("&"))
        order = []
        if "drinking" in parts:
            order.append("drinking")
        if "drugs" in parts:
            order.append("drugs")
        return [f"Yes-{'&'.join(order)}"]
    return lst

def _postprocess_field(field: str, values: List[str]) -> List[str]:
    if field in {"OffSex", "VicSex"}:
        return _collapse_sex_list(values)
    if field == "CoDefAccNum":
        return _collapse_codefaccnum_list(values)
    if field in {"OffIntoxOffence", "VicIntoxOffence"}:
        return _collapse_intox_list(values)
    return values

# -----------------------------
# Main conversion
# -----------------------------
def convert_output1_to_output2(output1: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    for field in FIELDS:
        raw_list = _ensure_list(output1.get(field))
        cleaned = [_normalize_sentinel_value(field, v) for v in raw_list]
        cleaned = [v for v in cleaned if v is not None and str(v).strip()]
        if not cleaned:
            cleaned = [DNA]

        conv = FIELD_CONVERTERS.get(field)

        conv_out: List[str] = []
        if conv is None:
            conv_out = cleaned
        else:
            for v in cleaned:
                if v is None or not str(v).strip():
                    continue
                # converter itself preserves DNA/DK if passed through
                res = conv(v)

                if res is None:
                    continue
                if isinstance(res, list):
                    for r in res:
                        if r is None or not str(r).strip():
                            continue
                        conv_out.append(str(r))
                else:
                    conv_out.append(str(res))

        # Normalize casing for sentinels after conversion
        conv_out = [_normalize_sentinel_value(field, v) for v in conv_out if v and str(v).strip()]

        # If everything got dropped, backfill DNA
        if not conv_out:
            conv_out = [DNA]

        # Important: DO NOT collapse-to-DNA if there are real values too.
        # Only make DNA-only if all are DNA.
        if all(_is_dna(v) for v in conv_out):
            conv_out = [DNA]
        else:
            conv_out = [v for v in conv_out if not _is_dna(v)]
            if not conv_out:
                conv_out = [DNA]

        # de-dupe + postprocess
        conv_out = _dedupe_keep_order(conv_out)
        conv_out = _postprocess_field(field, conv_out)

        out[field] = conv_out

    return out
