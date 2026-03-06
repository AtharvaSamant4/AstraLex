"""
intent_classifier.py — Lightweight intent detection for non-legal queries.

Classifies user input into:
  • ``chitchat``  — greetings, farewells, thanks, small-talk, meta questions
  • ``off_topic`` — general-knowledge questions unrelated to Indian law
  • ``legal``     — anything that should go through the RAG pipeline

Uses keyword / regex matching only — no LLM call required.
The pipeline uses the intent to decide whether to run the full RAG
stages or a lightweight conversational Gemini call instead.
"""

from __future__ import annotations

import re

# ── Pattern groups ──────────────────────────────────────────────────────────

_GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^h+e+l+o+!*$",          # hello, hellooo
        r"^hi+!*$",                # hi, hii, hiii
        r"^hey+!*$",               # hey, heyy
        r"^yo+!*$",                # yo
        r"^howdy!*$",
        r"^hola!*$",
        r"^namaste!*$",
        r"^namaskar!*$",
        r"^namaskara!*$",
        r"^greetings!*$",
        r"^sup!*\??$",
        r"^what'?s?\s*up!*\??$",
        r"^good\s*(morning|afternoon|evening|night|day)!*$",
        r"^(gm|gn)!*$",
        # Conversational / small-talk
        r"^how\s+are\s+you\??!*$",
        r"^how\s+are\s+you\s+doing\??!*$",
        r"^how'?s?\s+it\s+going\??!*$",
        r"^how\s+do\s+you\s+do\??!*$",
        r"^how\s+have\s+you\s+been\??!*$",
        r"^how'?s?\s+(your|the)\s+day\??!*$",
        r"^how'?s?\s+everything\??!*$",
        r"^how'?s?\s+life\??!*$",
        r"^i'?m\s+(good|fine|great|okay|ok|well|doing\s+well)!*$",
        r"^(all\s+)?good!*$",
        r"^nice\s+to\s+meet\s+you!*$",
        r"^pleased\s+to\s+meet\s+you!*$",
        r"^what'?s?\s+(good|new|happening|crackin|cracking)\??!*$",
        r"^long\s+time\s+no\s+see!*$",
        r"^kaise\s+ho\??!*$",          # Hindi: how are you?
        r"^kya\s+haal\s+hai\??!*$",    # Hindi: how are you?
        r"^aap\s+kaise\s+hain\??!*$",  # Hindi: how are you (formal)?
        r"^sab\s+badhiya\??!*$",       # Hindi: all good?
    )
]

_FAREWELL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^bye+!*$",
        r"^good\s*bye!*$",
        r"^see\s*y(ou|a)!*$",
        r"^take\s*care!*$",
        r"^(cya|ttyl|gtg)!*$",
        r"^adios!*$",
        r"^later!*$",
    )
]

_THANKS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^thanks?!*$",
        r"^thank\s*you!*$",
        r"^ty!*$",
        r"^thx!*$",
        r"^thanks?\s*a\s*lot!*$",
        r"^much\s*appreciated!*$",
        r"^dhanyavaad!*$",
        r"^shukriya!*$",
    )
]

_META_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^who\s+are\s+you\??$",
        r"^what\s+are\s+you\??$",
        r"^what\s+can\s+you\s+do\??$",
        r"^how\s+do\s+you\s+work\??$",
        r"^are\s+you\s+a\s+(bot|robot|ai|chatbot)\??$",
        r"^help!*\??$",
    )
]

# ── Legal keyword detection ────────────────────────────────────────────────
# If ANY of these words/phrases appear in the query, treat it as potentially
# legal even if the surface form looks like a general question.

_LEGAL_KEYWORDS: re.Pattern[str] = re.compile(
    r"""(?ix)                       # case-insensitive, verbose
    \b(?:
        # Acts & codes
        ipc | crpc | cr\.p\.c | i\.p\.c
        | indian\s+penal\s+code
        | code\s+of\s+criminal\s+procedure
        | constitution
        | fundamental\s+rights?
        | directive\s+principles?
        | hindu\s+marriage\s+act | hma
        | special\s+marriage\s+act | sma
        | dowry\s+prohibition | dowry\s+act
        | domestic\s+violence\s+act
        | protection\s+of\s+women
        | bharatiya\s+nyaya\s+sanhita | bns
        | bharatiya\s+nagarik\s+suraksha\s+sanhita | bnss

        # Legal concepts
        | section\s+\d+
        | article\s+\d+
        | bail | fir | chargesheet | charge\s*sheet
        | cognizable | non[- ]?cognizable | bailable | non[- ]?bailable
        | summons | warrant | arrest | custody | remand
        | murder | culpable\s+homicide | homicide | manslaughter
        | theft | robbery | dacoity | extortion | kidnapping
        | cheating | forgery | defamation | sedition | treason
        | assault | battery | grievous\s+hurt | hurt
        | dowry | streedhan | stridhan | mehr | mahr
        | divorce | maintenance | alimony | custody\s+of\s+child
        | marriage | nikah | void\s+marriage | voidable
        | adultery | bigamy | cruelty | desertion
        | abetment | conspiracy | attempt\s+to\s+commit
        | punishment | penalty | imprisonment | fine\s+under
        | offence | offense | criminal | penal
        | court | magistrate | sessions | high\s+court | supreme\s+court
        | advocate | lawyer | vakil | pleader
        | appeal | revision | review | writ | habeas\s+corpus
        | certiorari | mandamus | prohibition | quo\s+warranto
        | evidence | witness | testimony | oath | affidavit
        | complaint | petition | suit | plaint
        | accused | defendant | plaintiff | prosecution
        | acquittal | conviction | sentence | verdict
        | pardon | probation | commutation
        | preamble | amendment | schedule | proviso
        | right\s+to\s+equality | right\s+to\s+life
        | freedom\s+of\s+speech | freedom\s+of\s+religion
        | right\s+against\s+exploitation
        | right\s+to\s+education
        | legal | lawful | unlawful | illegal
        | judicial | jurisdiction | cognizance
        | anticipatory\s+bail | regular\s+bail
        | police | investigation | search\s+warrant
        | cyber\s*crime | it\s+act | information\s+technology\s+act
        | consumer\s+protection | negligence
        | tort | damages | compensation | liability
        | property | succession | inheritance | will\b | testament

        # Action verbs indicating criminal/legal scenarios
        | kills? | killing
        | steals? | stealing | stolen
        | robs? | robbing | robbed
        | attacks? | attacking
        | abducts? | abducting | abducted
        | kidnaps? | kidnapping
        | rapes? | raping | raped
        | molests? | molesting | molested
        | harasses? | harassing | harassed | harassment
        | threatens? | threatening | threatened
        | injures? | injuring | injured
        | harms? | harming | harmed
        | commits?\s+(?:a\s+)?(?:crime|offence|offense|fraud)
        | breaks?\s+(?:the\s+)?law
        | violat(?:es?|ing|ion) | violence

        # Scenario phrasing that implies legal questions
        | what\s+(?:is\s+the\s+)?(?:law|punishment|penalty|sentence)
        | what\s+happens?\s+(?:if|when)
        | is\s+it\s+(?:legal|illegal|a\s+crime|an?\s+offen[cs]e)
        | can\s+(?:i|you|someone|a\s+person)\s+be\s+(?:arrested|punished|jailed|charged|prosecuted)
        | what\s+(?:law|act|section|article)\s+(?:applies|governs|deals|covers)
        | under\s+(?:indian|criminal|the)\s+law
        | according\s+to\s+(?:indian\s+)?law
        | (?:indian|criminal)\s+law
        | crime | criminal\s+act
    )\b
    """
)

# ── Public API ──────────────────────────────────────────────────────────────

class IntentResult:
    """Result of intent classification."""

    __slots__ = ("is_legal", "intent")

    def __init__(self, is_legal: bool, intent: str) -> None:
        self.is_legal = is_legal
        self.intent = intent


def _has_legal_keywords(text: str) -> bool:
    """Return True if *text* contains any recognisable legal term."""
    return bool(_LEGAL_KEYWORDS.search(text))


def classify_intent(text: str) -> IntentResult:
    """
    Classify whether *text* is chitchat, off-topic, or a legal question.

    Returns an ``IntentResult``.  If ``result.is_legal`` is ``False``,
    the caller should route the query to a lightweight conversational
    Gemini call instead of the full RAG pipeline.
    """
    cleaned = text.strip()

    # Remove trailing punctuation for matching, but keep original
    stripped = re.sub(r"[.!?,;:\s]+$", "", cleaned)

    if not stripped:
        return IntentResult(False, "empty")

    # ── Chitchat patterns (exact match) ────────────────────────────────────
    for pat in _GREETING_PATTERNS:
        if pat.match(stripped):
            return IntentResult(False, "greeting")

    for pat in _FAREWELL_PATTERNS:
        if pat.match(stripped):
            return IntentResult(False, "farewell")

    for pat in _THANKS_PATTERNS:
        if pat.match(stripped):
            return IntentResult(False, "thanks")

    for pat in _META_PATTERNS:
        if pat.match(stripped):
            return IntentResult(False, "meta")

    # ── Off-topic detection ────────────────────────────────────────────────
    # If the query contains legal keywords → always let it through.
    # Otherwise, flag it as off-topic so the expensive pipeline is skipped.
    if not _has_legal_keywords(cleaned):
        return IntentResult(False, "off_topic")

    return IntentResult(True, "legal")
