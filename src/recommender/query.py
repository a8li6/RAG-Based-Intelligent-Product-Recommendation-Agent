"""Understand a shopping request without guessing.

Explicit amounts and units are parsed with narrow, documented English patterns. Anything
ambiguous, unsupported or contradictory becomes a clarification question; a hard
constraint is never invented. Structured controls are authoritative for their field.

Documented interpretation rules:
- "under/below/less than $900" -> max 89999 cents; "at most/up to/budget $900" -> 90000.
- Weight limits are inclusive: "under 1.5 kg" is treated as "at most 1.5 kg".
- A bare "16 GB RAM" (no direction) stays a soft preference; "at least 16 GB RAM" is hard.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field

from recommender.schemas import (
    MAX_PRICE_CENTS,
    MAX_RAM_GB,
    MAX_WEIGHT_KG,
    Constraints,
    Filters,
)

MAX_HISTORY = 5

NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MONEY = rf"(?:\$\s*{NUM}|{NUM}\s*(?:usd|dollars?|bucks))"
STRICT = r"(?:under|below|less than|cheaper than|lower than|beneath)"
INCLUSIVE = (
    r"(?:at most|up to|max(?:imum)?(?:\s+of)?|no more than|not more than|not exceeding"
    r"|within|budget(?:\s+(?:of|is))?)"
)
KG = r"(?:kg|kgs|kilograms?|kilos?)"
GB = r"(?:gb|gigabytes?)"
RAM_WORD = r"(?:ram|memory)"
FLAGS = re.IGNORECASE

PRICE_PRE = re.compile(
    rf"(?:\bbudget\s+)?\b(?P<cue>{STRICT}|{INCLUSIVE})\s*(?:a\s+)?(?P<money>{MONEY})", FLAGS
)
PRICE_POST = re.compile(
    rf"(?P<money>{MONEY})\s*(?P<post>or less|or under|or below|max(?:imum)?|budget|limit|tops)\b",
    FLAGS,
)
PRICE_MIN = re.compile(
    rf"\b(?:over|above|(?<!no )(?<!not )more than|at least|minimum|min|starting at|from|greater than)\s*"
    rf"(?:a\s+)?{MONEY}",
    FLAGS,
)
PRICE_BETWEEN = re.compile(rf"\bbetween\s*{MONEY}\s*(?:and|to|-)\s*{MONEY}", FLAGS)
NEGATIVE_MONEY = re.compile(r"\$\s*[-−]\s*\d|[-−]\s*\$\s*\d", FLAGS)
FOREIGN_CURRENCY = re.compile(
    r"[€£¥₹₩]\s*\d"
    r"|\b\d[\d,.]*\s*(?:eur|euros?|gbp|jpy|yen|cad|aud|inr|rials?|tomans?)\b"
    r"|\b(?:eur|gbp)\s*\d",
    FLAGS,
)

RAM_MIN = [
    re.compile(
        rf"\b(?:at least|minimum(?:\s+of)?|min|no less than|not less than)\s*(?P<n>{NUM})\s*"
        rf"{GB}\s*(?:of\s+)?{RAM_WORD}\b",
        FLAGS,
    ),
    re.compile(
        rf"\b{RAM_WORD}\s*(?:of\s*)?(?:at least|minimum(?:\s+of)?|min)\s*(?P<n>{NUM})\s*{GB}\b",
        FLAGS,
    ),
    re.compile(
        rf"\b(?P<n>{NUM})\s*{GB}\s*(?:of\s+)?{RAM_WORD}\s*"
        r"(?:or more|or higher|or above|or better|minimum)\b",
        FLAGS,
    ),
    re.compile(rf"\b(?P<n>{NUM})\+\s*{GB}\s*(?:of\s+)?{RAM_WORD}\b", FLAGS),
]
RAM_MAX = re.compile(
    rf"\b(?:at most|up to|max(?:imum)?(?:\s+of)?|no more than|under|below|less than)\s*"
    rf"(?P<n>{NUM})\s*{GB}\s*(?:of\s+)?{RAM_WORD}\b",
    FLAGS,
)

WEIGHT_MAX = [
    re.compile(
        rf"(?:\bweight\s+(?:of\s+)?)?\b(?:{STRICT}|at most|up to|max(?:imum)?(?:\s+of)?"
        rf"|no more than|not more than"
        rf"|not exceeding|lighter than|weigh(?:ing|s)?\s+(?:less than|under|at most|up to))\s*"
        rf"(?P<n>{NUM})\s*{KG}\b",
        FLAGS,
    ),
    re.compile(
        rf"(?P<n>{NUM})\s*{KG}\s*(?:or less|or lighter|or under|or below|max(?:imum)?|tops)\b",
        FLAGS,
    ),
]
WEIGHT_MIN = re.compile(
    rf"\b(?:at least|over|above|(?<!no )(?<!not )more than|heavier than|minimum)\s*(?P<n>{NUM})\s*{KG}\b",
    FLAGS,
)
WEIGHT_BAD_UNIT = re.compile(rf"\b{NUM}\s*(?:lbs?|pounds?|oz|ounces?|grams?)\b", FLAGS)

IN_STOCK = re.compile(r"\bin[- ]stock\b", FLAGS)
SOFT_PRICE = re.compile(
    r"\b(?:cheap(?:est)?|inexpensive|affordable|budget[- ]friendly|low[- ]cost"
    r"|not too expensive)\b",
    FLAGS,
)
CLEAR = [
    re.compile(
        r"\b(?:forget|remove|drop|clear|ignore|cancel)\s+(?:about\s+)?(?:the\s+)?"
        r"(?P<what>weight|budget|price|ram|memory)\b"
        r"(?:\s+(?:limit|constraint|requirement|cap|minimum|maximum|filter))?",
        FLAGS,
    ),
    re.compile(
        r"\bno\s+(?P<what>weight|budget|price|ram|memory)\s+"
        r"(?:limit|constraint|requirement|cap|filter)\b",
        FLAGS,
    ),
]
CLEAR_FIELD = {
    "weight": "max_weight_kg",
    "budget": "max_price_cents",
    "price": "max_price_cents",
    "ram": "min_ram_gb",
    "memory": "min_ram_gb",
}
FIELD_LABEL = {
    "max_price_cents": "budget",
    "min_ram_gb": "minimum RAM",
    "max_weight_kg": "maximum weight",
}

NOISE = {
    "i", "need", "want", "looking", "for", "find", "me", "get", "show", "give", "a", "an",
    "the", "and", "but", "with", "that", "which", "also", "plus", "please", "only", "instead",
    "now", "then", "actually", "is", "are", "it", "should", "be", "at", "of", "to", "has",
    "have", "must", "under", "below", "within",
}  # fmt: skip
DEFAULT_SEMANTIC = "laptop"


class InvalidQueryError(ValueError):
    """The raw query is empty or longer than the configured limit."""


class QueryIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_query: str
    max_price_cents: int | None = None
    currency: str = "USD"
    min_ram_gb: int | None = None
    max_weight_kg: float | None = None
    require_in_stock: bool = True
    requested_count: int = Field(default=3, ge=1, le=3)
    needs_clarification: bool = False
    clarification_question: str | None = None
    explicit_fields: list[str] = Field(default_factory=list)  # provenance for session updates

    def constraints(self) -> Constraints:
        return Constraints(
            max_price_cents=self.max_price_cents,
            min_ram_gb=self.min_ram_gb,
            max_weight_kg=self.max_weight_kg,
            require_in_stock=self.require_in_stock,
        )


class SessionState(BaseModel):
    """Structured active filters, recent semantic intent and a bounded short history."""

    model_config = ConfigDict(extra="forbid")

    max_price_cents: int | None = None
    min_ram_gb: int | None = None
    max_weight_kg: float | None = None
    require_in_stock: bool = True
    semantic_query: str | None = None
    history: list[str] = Field(default_factory=list)


class ParseResult(BaseModel):
    intent: QueryIntent
    session: SessionState


class _Parsed:
    def __init__(self) -> None:
        self.values: dict[str, list] = {
            "max_price_cents": [],
            "min_ram_gb": [],
            "max_weight_kg": [],
        }
        self.cleared: set[str] = set()
        self.stock: bool | None = None
        self.spans: list[tuple[int, int]] = []
        self.question: str | None = None
        self.semantic: str | None = None

    def ask(self, question: str) -> None:
        if self.question is None:
            self.question = question


def normalize_query(text: str, max_chars: int = 2000) -> str:
    """NFKC-normalize, drop control characters, collapse whitespace, bound the length."""
    if not isinstance(text, str):
        raise InvalidQueryError("query must be text")
    cleaned = "".join(
        ch if (ch.isspace() or unicodedata.category(ch)[0] != "C") else " "
        for ch in unicodedata.normalize("NFKC", text)
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise InvalidQueryError("query must not be empty")
    if len(cleaned) > max_chars:
        raise InvalidQueryError(f"query is longer than {max_chars} characters")
    return cleaned


def _decimal(token: str) -> Decimal:
    return Decimal(re.search(NUM, token).group(0).replace(",", ""))  # type: ignore[union-attr]


def _money_cents(token: str, strict: bool) -> int | str:
    """Cents for a money token, or a clarification message."""
    try:
        cents = _decimal(token) * 100
    except InvalidOperation:
        return "I could not read that amount. Please state the budget as, for example, $900."
    if cents != cents.to_integral_value():
        return "Please give the budget with at most two decimal places (whole cents)."
    value = int(cents) - (1 if strict else 0)
    if value < 0:
        return "That budget leaves no possible price. Please give a positive budget."
    if value > MAX_PRICE_CENTS:
        return "That budget is outside the supported range. Please check the amount."
    return value


def _parse_text(text: str) -> _Parsed:
    p = _Parsed()

    for pattern, question in (
        (NEGATIVE_MONEY, "Amounts cannot be negative. What is your maximum budget in USD?"),
        (FOREIGN_CURRENCY, "Only USD is supported. Please give the budget in US dollars."),
        (PRICE_BETWEEN, "I can only apply a maximum budget. What is the most you want to spend?"),
        (
            PRICE_MIN,
            "I can only apply a maximum budget, not a minimum price. What is your maximum?",
        ),
        (RAM_MAX, "I can only filter by a minimum amount of RAM. What minimum RAM do you need?"),
        (
            WEIGHT_MIN,
            "I can only filter by a maximum weight. What is the heaviest you would accept?",
        ),
        (WEIGHT_BAD_UNIT, "Please give weight in kilograms, for example 'at most 1.5 kg'."),
    ):
        for m in pattern.finditer(text):
            p.ask(question)
            p.spans.append(m.span())

    for m in PRICE_PRE.finditer(text):
        strict = re.fullmatch(STRICT, m.group("cue"), FLAGS) is not None
        _record_price(p, m, strict)
    for m in PRICE_POST.finditer(text):
        if not _overlaps(p.spans, m.span()):
            _record_price(p, m, strict=False)

    for pattern in RAM_MIN:
        for m in pattern.finditer(text):
            if _overlaps(p.spans, m.span()):
                continue
            number = _decimal(m.group("n"))
            if number != number.to_integral_value() or not 1 <= number <= MAX_RAM_GB:
                p.ask("Please give the minimum RAM as a whole number of GB, for example 16.")
            else:
                p.values["min_ram_gb"].append(int(number))
            p.spans.append(m.span())

    for pattern in WEIGHT_MAX:
        for m in pattern.finditer(text):
            if _overlaps(p.spans, m.span()):
                continue
            number = _decimal(m.group("n"))
            if not 0 < number <= Decimal(str(MAX_WEIGHT_KG)):
                p.ask("Please give a maximum weight in kilograms, for example 1.5.")
            else:
                p.values["max_weight_kg"].append(float(number))
            p.spans.append(m.span())

    for m in IN_STOCK.finditer(text):
        p.stock = True
        p.spans.append(m.span())

    for pattern in CLEAR:
        for m in pattern.finditer(text):
            p.cleared.add(CLEAR_FIELD[m.group("what").lower()])
            p.spans.append(m.span())

    remaining = _blank(text, p.spans)
    if re.search(MONEY, remaining, FLAGS):
        p.ask(
            "I found an amount but cannot tell what it limits. "
            "Did you mean a maximum budget, e.g. 'under $900' or 'at most $900'?"
        )

    for field, found in p.values.items():
        if len(set(found)) > 1:
            p.ask(f"Your request gives conflicting values for the {FIELD_LABEL[field]}. Which one?")

    p.semantic = _semantic_text(remaining)
    return p


def _record_price(p: _Parsed, m: re.Match, strict: bool) -> None:
    result = _money_cents(m.group("money"), strict)
    if isinstance(result, str):
        p.ask(result)
    else:
        p.values["max_price_cents"].append(result)
    p.spans.append(m.span())


def _overlaps(spans: list[tuple[int, int]], span: tuple[int, int]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in spans)


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for i in range(start, end):
            chars[i] = " "
    return "".join(chars)


def _semantic_text(remaining: str) -> str | None:
    pieces = []
    for piece in re.split(r"[,;]", remaining):
        words = [w for w in re.split(r"\s+", piece.strip(" .:!?")) if w]
        while words and words[0].lower().strip(".:!?") in NOISE:
            words.pop(0)
        while words and words[-1].lower().strip(".:!?") in NOISE:
            words.pop()
        if words:
            pieces.append(" ".join(words))
    text = ", ".join(pieces)
    return text if re.search(r"\w", text) else None


def parse_and_merge(
    text: str,
    controls: Filters | None = None,
    session: SessionState | None = None,
    *,
    requested_count: int = 3,
    max_chars: int = 2000,
) -> ParseResult:
    """Parse one request and merge it with structured controls and session state.

    The returned session is updated only when the request is fully understood.
    """
    session = session or SessionState()
    controls = controls or Filters()
    normalized = normalize_query(text, max_chars)
    parsed = _parse_text(normalized)

    merged = {
        "max_price_cents": session.max_price_cents,
        "min_ram_gb": session.min_ram_gb,
        "max_weight_kg": session.max_weight_kg,
    }
    stock = session.require_in_stock
    explicit: list[str] = []
    question = parsed.question

    for field in parsed.cleared:
        merged[field] = None
        explicit.append(field)
    for field, found in parsed.values.items():
        if found:
            merged[field] = found[0]
            explicit.append(field)
    if parsed.stock is not None:
        stock = parsed.stock
        explicit.append("require_in_stock")

    control_values = {
        "max_price_cents": controls.max_price_cents,
        "min_ram_gb": controls.min_ram_gb,
        "max_weight_kg": controls.max_weight_kg,
    }
    for field, control in control_values.items():
        if control is None:
            continue
        text_value = parsed.values[field][0] if parsed.values[field] else None
        if (text_value is not None and text_value != control) or field in parsed.cleared:
            shown = "cleared" if text_value is None else text_value
            question = question or (
                f"The {FIELD_LABEL[field]} in your text ({shown}) conflicts with the "
                f"control value ({control}). Which should I use?"
            )
        merged[field] = control
        if field not in explicit:
            explicit.append(field)
    if controls.require_in_stock is not None:
        if parsed.stock is not None and parsed.stock != controls.require_in_stock:
            question = question or "Your text and the stock control disagree. Which should I use?"
        stock = controls.require_in_stock
        if "require_in_stock" not in explicit:
            explicit.append("require_in_stock")

    semantic = parsed.semantic or session.semantic_query or DEFAULT_SEMANTIC
    if question is None and merged["max_price_cents"] is None and SOFT_PRICE.search(normalized):
        question = "What is your maximum budget in USD? I will not guess a price limit."

    intent = QueryIntent(
        semantic_query=semantic,
        max_price_cents=merged["max_price_cents"],
        min_ram_gb=merged["min_ram_gb"],
        max_weight_kg=merged["max_weight_kg"],
        require_in_stock=stock,
        requested_count=requested_count,
        needs_clarification=question is not None,
        clarification_question=question,
        explicit_fields=sorted(explicit),
    )
    if question is not None:
        return ParseResult(intent=intent, session=session)

    new_session = SessionState(
        max_price_cents=merged["max_price_cents"],
        min_ram_gb=merged["min_ram_gb"],
        max_weight_kg=merged["max_weight_kg"],
        require_in_stock=stock,
        semantic_query=semantic,
        history=[*session.history, semantic][-MAX_HISTORY:],
    )
    return ParseResult(intent=intent, session=new_session)
