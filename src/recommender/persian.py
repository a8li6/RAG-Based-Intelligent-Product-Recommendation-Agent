"""Persian (Farsi) support for query understanding: normalization, constraint phrases, lexicon.

Scope: Persian QUERIES over the English catalog. Responses stay English. Persian concept words map
to the SAME concept identifiers as the English lexicon, so a Persian request can match English
catalog documents without translating the catalog. Words that are not in the lexicon contribute
no retrieval features (a Persian character n-gram could never match an English document).

Like the English lexicon this is hand-written, small and transparent. It is not a language model:
it does not understand negation, morphology beyond a few common suffixes, or words it does not list.
"""

from __future__ import annotations

import re

NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
PERSIAN_LETTERS = re.compile(r"[؀-ۿﭐ-﷿ﹰ-﻿]")

_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
_LETTERS = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "ـ": ""})
_PUNCT = str.maketrans({"٫": ".", "٬": ",", "،": ", ", "؛": "; ", "؟": "? "})
_SLASH_DECIMAL = re.compile(r"(\d)\s*/\s*(\d)(?=\s*(?:کیلو|kg))")


def has_persian(text: str) -> bool:
    return bool(PERSIAN_LETTERS.search(text))


def normalize_persian(text: str) -> str:
    """Map Persian/Arabic digits and separators to ASCII, unify Arabic letter variants.

    The zero-width non-joiner is handled by the caller's control-character cleanup (it becomes
    a space), so "لپ‌تاپ" and "لپ تاپ" end up identical.
    """
    text = text.translate(_DIGITS).translate(_LETTERS).translate(_PUNCT)
    return _SLASH_DECIMAL.sub(r"\1.\2", text)


# --- constraint phrases (applied to already-normalized text) -------------------------------
_GB = r"(?:گیگابایت|گیگا\s*بایت|گیگ|گیگا|gb)"
_RAM = r"(?:\bرم\b|\bram\b|حافظه\s*(?:ی\s*)?رم\b)"
_KG = r"(?:کیلوگرم|کیلو|kg)"
MONEY = rf"(?:{NUM}\s*(?:دلار(?:ی)?|usd)|\$\s*{NUM})"
_STRICT = r"(?:زیر|کمتر\s+از|پایین\s?تر\s+از|ارزان\s?تر\s+از)"
_INCL = (
    r"(?:حداکثر|حد\s?اکثر|نهایتا|نهایتاً|سقف|\bتا\b|بودجه(?:\s*(?:ام|م|من))?(?:\s*(?:حدود|حداکثر))?)"
)
_MORE = r"(?:یا\s*(?:بیشتر|بالاتر|بالا)|به\s*بالا|و\s*(?:بیشتر|بالاتر))"

PRICE_PRE = re.compile(rf"(?:\b(?P<strict>{_STRICT})|(?P<incl>{_INCL}))\s*(?P<money>{MONEY})")
PRICE_POST = re.compile(
    rf"(?P<money>{MONEY})\s*(?P<post>یا\s*(?:کمتر|زیر|پایین\s?تر)|حداکثر|سقف|بودجه)"
)
PRICE_MIN = re.compile(rf"\b(?:بالای|بیشتر\s+از|بالاتر\s+از|حداقل)\s*{MONEY}")
PRICE_BETWEEN = re.compile(
    rf"\b(?:بین|از)\s*{NUM}\s*(?:دلار)?\s*(?:تا|و|الی)\s*{NUM}\s*(?:دلار(?:ی)?|usd)"
)
FOREIGN_CURRENCY = re.compile(
    rf"{NUM}\s*(?:(?:میلیون|هزار|میلیارد)\s*)?(?:تومان|تومن|ریال|یورو|پوند|لیر|درهم)"
)

RAM_MIN = [
    re.compile(rf"(?:حداقل|دست\s*کم)\s*(?P<n>{NUM})\s*{_GB}\s*{_RAM}"),
    re.compile(rf"{_RAM}\s*(?:حداقل|دست\s*کم)\s*(?P<n>{NUM})\s*{_GB}"),
    re.compile(rf"(?P<n>{NUM})\s*{_GB}\s*{_RAM}\s*{_MORE}"),
    re.compile(rf"{_RAM}\s*(?P<n>{NUM})\s*{_GB}\s*{_MORE}"),
]
RAM_MAX = re.compile(rf"(?:حداکثر|کمتر\s+از|زیر|\bتا\b)\s*(?P<n>{NUM})\s*{_GB}\s*{_RAM}")

WEIGHT_MAX = [
    re.compile(
        rf"(?:وزن\s*)?(?:{_STRICT}|سبک\s?تر\s+از|حداکثر|حد\s?اکثر|نهایتا|نهایتاً|سقف|\bتا\b)\s*"
        rf"(?P<n>{NUM})\s*{_KG}"
    ),
    re.compile(rf"(?P<n>{NUM})\s*{_KG}\s*(?:یا\s*(?:کمتر|سبک\s?تر|زیر|پایین\s?تر)|حداکثر)"),
]
WEIGHT_MIN = re.compile(rf"(?:حداقل|بیشتر\s+از|سنگین\s?تر\s+از|بالای)\s*(?P<n>{NUM})\s*{_KG}")
WEIGHT_BAD_UNIT = re.compile(rf"\b{NUM}\s*(?:پوند|اونس|گرم)\b")

IN_STOCK = re.compile(r"\bموجود(?:\s+در\s+انبار)?\b")
SOFT_PRICE = re.compile(r"\b(?:ارزان(?:\s?قیمت)?|اقتصادی|کم\s?هزینه|ارزان\s?ترین)\b")
CLEAR = [
    re.compile(
        r"محدودیت\s*(?P<what>وزن|بودجه|قیمت|رم)\s*(?:را\s*)?(?:فراموش\s*کن|حذف\s*کن|بردار|نادیده\s*بگیر)"
    ),
    re.compile(r"(?:بدون|بی)\s*محدودیت\s*(?P<what>وزن|بودجه|قیمت|رم)"),
]
CLEAR_FIELD = {
    "وزن": "max_weight_kg",
    "بودجه": "max_price_cents",
    "قیمت": "max_price_cents",
    "رم": "min_ram_gb",
}

# Words that carry no retrieval meaning at the edges of the remaining request text.
NOISE = frozenset(
    [
        "برای",
        "یک",
        "و",
        "با",
        "از",
        "در",
        "که",
        "را",
        "می",
        "خواهم",
        "میخواهم",
        "میخوام",
        "نیاز",
        "دارم",
        "لازم",
        "است",
        "هست",
        "باشه",
        "باشد",
        "لطفا",
        "فقط",
        "هم",
        "یا",
        "به",
        "این",
        "آن",
        "من",
        "ما",
        "چیزی",
        "دنبال",
        "هستم",
        "هستیم",
        "میشه",
        "بده",
        "پیدا",
        "کن",
        "ببینم",
        "ببین",
        "لپ",
        "تاپ",
        "لپتاپ",
    ]
)


# --- concept lexicon: Persian phrases -> the SAME concept ids as the English lexicon ---------
CONCEPTS: dict[str, tuple[str, ...]] = {
    "concept_code": (
        "برنامه نویسی",
        "برنامه نویس",
        "کدنویسی",
        "کد نویسی",
        "کد زنی",
        "توسعه نرم افزار",
        "توسعه دهنده",
        "توسعه",
        "نرم افزار",
        "پایتون",
        "جاوا",
        "دولوپر",
        "دولوپمنت",
        "پروگرمینگ",
    ),
    "concept_travel": (
        "سفر",
        "مسافرت",
        "مسافر",
        "قابل حمل",
        "قابل جابجایی",
        "سبک",
        "کم وزن",
        "جمع و جور",
        "همراه",
        "حمل",
        "پرواز",
        "رفت و آمد",
        "رفت وآمد",
        "رفتوآمد",
        "مترو",
        "اتوبوس",
        "ایاب و ذهاب",
    ),
    "concept_study": (
        "دانشجو",
        "دانشگاه",
        "درس",
        "کلاس",
        "مدرسه",
        "تکلیف",
        "تکالیف",
        "یادگیری",
        "دانش آموز",
        "تحصیل",
        "مطالعه",
        "امتحان",
    ),
    "concept_gaming": ("بازی", "گیم", "گیمینگ", "گیمر", "اسپورت"),
    "concept_creative": (
        "ویدیو",
        "ویدئو",
        "تدوین",
        "ادیت",
        "رندر",
        "رندرینگ",
        "طراحی گرافیک",
        "گرافیست",
        "عکاسی",
        "انیمیشن",
        "تولید محتوا",
        "محتوا",
        "سه بعدی",
        "تری دی",
        "مدل سازی",
        "خلاقانه",
        "خلاق",
        "طراح",
    ),
    "concept_office": (
        "اداری",
        "اداره",
        "دفتر",
        "دورکاری",
        "دور کاری",
        "اکسل",
        "ایمیل",
        "اسناد",
        "ارائه",
        "پرزنتیشن",
        "میز کار",
        "خانه",
        "منزل",
        "نشیمن",
        "بیزنس",
        "تجاری",
    ),
    "concept_browse": ("وب", "اینترنت", "مرور", "فیلم", "آنلاین"),
    "concept_heavy": ("قدرتمند", "قوی", "حرفه ای"),
}
# A few common suffixes are allowed after a phrase (plural, "-i", possessives).
_SUFFIX = r"(?:\s?(?:ها|های|هایی)|ی|ای|ام|م|تان|مان|ش)?"
_LEFT = r"(?<![؀-ۿ])"
_RIGHT = r"(?![؀-ۿ])"
_PATTERNS = {
    concept: re.compile(
        _LEFT + "(?:" + "|".join(re.escape(p) for p in sorted(words, key=len, reverse=True)) + ")"
        + _SUFFIX + _RIGHT
    )
    for concept, words in CONCEPTS.items()
}  # fmt: skip


def concepts_of(text: str) -> frozenset[str]:
    """Concept ids triggered by the Persian words of a text (empty for non-Persian text)."""
    if not has_persian(text):
        return frozenset()
    normalized = re.sub(r"[‌‍]", " ", normalize_persian(text))
    return frozenset(c for c, pattern in _PATTERNS.items() if pattern.search(normalized))
