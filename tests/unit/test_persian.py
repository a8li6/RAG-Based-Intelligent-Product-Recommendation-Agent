import re

import pytest

from recommender import persian as fa
from recommender.local_embedder import CONCEPTS as EN_CONCEPTS

# --- normalization -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("۰۱۲۳۴۵۶۷۸۹", "0123456789"),
        ("٠١٢٣٤٥٦٧٨٩", "0123456789"),  # Arabic-Indic digits
        ("۱٫۵", "1.5"),  # Persian decimal separator
        ("۱٬۲۰۰", "1,200"),  # Persian thousands separator
        ("اول، دوم", "اول,  دوم"),  # Persian comma
        ("کجاست؟", "کجاست? "),
        ("نظیر", "نظیر"),
        ("كتاب يك", "کتاب یک"),  # Arabic kaf and yeh
        ("ســلام", "سلام"),  # tatweel removed
    ],
)
def test_normalization_maps_to_ascii_and_unifies_letters(raw, expected):
    assert fa.normalize_persian(raw) == expected


def test_slash_decimal_is_converted_only_before_a_kilo_unit():
    assert fa.normalize_persian("۱/۲ کیلو") == "1.2 کیلو"
    assert fa.normalize_persian("۱ / ۵ کیلوگرم") == "1.5 کیلوگرم"
    assert fa.normalize_persian("۱۶/۳۲ گیگ") == "16/32 گیگ"  # not a decimal here


def test_normalization_leaves_english_text_alone():
    assert (
        fa.normalize_persian("laptop under $900, at most 1.5 kg")
        == "laptop under $900, at most 1.5 kg"
    )


def test_script_detection():
    assert fa.has_persian("لپ‌تاپ") and fa.has_persian("buy لپ‌تاپ")
    assert not fa.has_persian("laptop 123")


# --- the lexicon shares concept ids with the English lexicon ------------------------------------


def test_every_persian_concept_exists_in_the_english_lexicon():
    assert set(fa.CONCEPTS) <= set(EN_CONCEPTS)


@pytest.mark.parametrize(
    ("text", "concept"),
    [
        ("برنامه‌نویسی", "concept_code"),
        ("برنامه نویس هستم", "concept_code"),
        ("کدنویسی", "concept_code"),
        ("یک سفر طولانی", "concept_travel"),
        ("کم‌وزن و قابل حمل", "concept_travel"),
        ("برای دانشجوها", "concept_study"),
        ("تکالیف مدرسه", "concept_study"),
        ("بازی‌های رایانه‌ای", "concept_gaming"),
        ("گیمینگ", "concept_gaming"),
        ("تدوین فیلم", "concept_creative"),
        ("طراحی گرافیک", "concept_creative"),
        ("رندر سه‌بعدی", "concept_creative"),
        ("دورکاری", "concept_office"),
        ("اکسل و ایمیل", "concept_office"),
        ("مرور اینترنت", "concept_browse"),
        ("دستگاه قدرتمند", "concept_heavy"),
    ],
)
def test_persian_words_trigger_the_shared_concepts(text, concept):
    assert concept in fa.concepts_of(text)


@pytest.mark.parametrize(
    "text",
    [
        "لپ‌تاپ",  # the product word itself carries no concept
        "کارت گرافیک",  # a graphics card is not graphic design
        "درست است",  # starts with the letters of "lesson" but is another word
        "بازیگر سینما",  # "actor", not "game"
        "کار",  # the generic word "work"
        "دستور پخت اسموتی موز",
        "hello world",
        "",
    ],
)
def test_unrelated_or_generic_words_trigger_nothing(text):
    assert fa.concepts_of(text) == frozenset()


def test_spacing_variants_give_the_same_result():
    joined = fa.concepts_of("برنامه‌نویسی")
    assert joined == fa.concepts_of("برنامه نویسی") == fa.concepts_of("برنامه‌نویسی")
    assert fa.concepts_of("لپ‌تاپ برای بازی") == fa.concepts_of("لپ تاپ برای بازی")


def test_arabic_letter_variants_match_too():
    assert "concept_study" in fa.concepts_of("دانشجو".replace("ی", "ي"))


# --- constraint patterns -------------------------------------------------------------------


def m(pattern, text):
    return pattern.search(fa.normalize_persian(text).replace("‌", " "))


def test_budget_patterns():
    assert m(fa.PRICE_PRE, "زیر ۷۵۰ دلار").group("strict")
    assert m(fa.PRICE_PRE, "کمتر از ۱٬۲۰۰ دلار").group("strict")
    assert m(fa.PRICE_PRE, "حداکثر ۵۰۰ دلار").group("incl")
    assert m(fa.PRICE_PRE, "تا ۸۰۰ دلار").group("incl")
    assert m(fa.PRICE_PRE, "بودجه‌ام ۶۵۰ دلار است").group("incl")
    assert m(fa.PRICE_POST, "۶۰۰ دلار یا کمتر")
    assert m(fa.PRICE_MIN, "بالای ۴۰۰ دلار")
    assert m(fa.PRICE_BETWEEN, "بین ۳۰۰ تا ۶۰۰ دلار")
    assert m(fa.PRICE_BETWEEN, "از ۳۰۰ تا ۶۰۰ دلار")


def test_foreign_currencies_are_detected():
    for text in ("۲۰ میلیون تومان", "۳۰۰ هزار تومن", "۵۰ یورو", "۱۰۰۰۰ ریال"):
        assert m(fa.FOREIGN_CURRENCY, text), text
    assert not m(fa.FOREIGN_CURRENCY, "۹۰۰ دلار")


def test_ram_patterns():
    assert m(fa.RAM_MIN[0], "حداقل ۸ گیگ رم").group("n") == "8"
    assert m(fa.RAM_MIN[0], "حداقل ۳۲ گیگابایت رم").group("n") == "32"
    assert m(fa.RAM_MIN[1], "رم حداقل ۱۶ گیگ").group("n") == "16"
    assert m(fa.RAM_MIN[2], "۱۶ گیگ رم یا بیشتر").group("n") == "16"
    assert m(fa.RAM_MIN[3], "رم ۱۶ گیگ یا بالاتر").group("n") == "16"
    assert m(fa.RAM_MAX, "حداکثر ۸ گیگ رم").group("n") == "8"


def test_ram_is_not_confused_with_longer_words():
    assert not m(fa.RAM_MIN[0], "حداقل ۸ گیگ رمان")  # "novel", not RAM


def test_weight_patterns():
    assert m(fa.WEIGHT_MAX[0], "حداکثر ۱٫۴ کیلوگرم").group("n") == "1.4"
    assert m(fa.WEIGHT_MAX[0], "وزن زیر ۲ کیلو").group("n") == "2"
    assert m(fa.WEIGHT_MAX[0], "سبک‌تر از ۱/۳ کیلو").group("n") == "1.3"
    assert m(fa.WEIGHT_MAX[1], "۱.۵ کیلو یا کمتر").group("n") == "1.5"
    assert m(fa.WEIGHT_MIN, "حداقل ۲ کیلو")
    assert m(fa.WEIGHT_BAD_UNIT, "۳ پوند") and m(fa.WEIGHT_BAD_UNIT, "۵۰۰ گرم")
    assert not m(fa.WEIGHT_BAD_UNIT, "۲ کیلوگرم")  # "kilogram" ends in "gram" but is supported


def test_stock_and_soft_price_words():
    assert m(fa.IN_STOCK, "فقط موجود در انبار")
    assert not m(fa.IN_STOCK, "ناموجود")
    assert m(fa.SOFT_PRICE, "یک لپ‌تاپ ارزان")
    assert m(fa.SOFT_PRICE, "اقتصادی")


def test_clear_phrases():
    assert m(fa.CLEAR[0], "محدودیت وزن را حذف کن").group("what") == "وزن"
    assert m(fa.CLEAR[1], "بدون محدودیت بودجه").group("what") == "بودجه"
    assert re.fullmatch(r"[؀-ۿ]+", "وزن") and fa.CLEAR_FIELD["رم"] == "min_ram_gb"
