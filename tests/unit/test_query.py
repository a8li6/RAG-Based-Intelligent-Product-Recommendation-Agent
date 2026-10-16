import pytest

from recommender.query import (
    InvalidQueryError,
    SessionState,
    normalize_query,
    parse_and_merge,
)
from recommender.schemas import Filters


def intent(text, **kwargs):
    return parse_and_merge(text, **kwargs).intent


# --- supported syntax ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "cents"),
    [
        ("laptop under $900", 89999),
        ("laptop below $900", 89999),
        ("laptop less than $900", 89999),
        ("laptop at most $900", 90000),
        ("laptop up to $900", 90000),
        ("laptop max $900", 90000),
        ("laptop no more than $900", 90000),
        ("laptop with a budget of $900", 90000),
        ("budget under $900 for coding", 89999),
        ("laptop for $900 or less", 90000),
        ("laptop, $900 budget", 90000),
        ("laptop under 900 dollars", 89999),
        ("laptop under 900 USD", 89999),
        ("laptop under $899.99", 89998),
        ("laptop at most $1,200", 120000),
        ("laptop at most $1,299.50", 129950),
        ("laptop UNDER $900", 89999),
    ],
)
def test_budget_patterns(text, cents):
    result = intent(text)
    assert result.max_price_cents == cents
    assert not result.needs_clarification


@pytest.mark.parametrize(
    ("text", "ram"),
    [
        ("at least 16 GB RAM", 16),
        ("at least 16GB of RAM", 16),
        ("minimum 16 GB RAM", 16),
        ("minimum of 8 gigabytes memory", 8),
        ("RAM of at least 32 GB", 32),
        ("16 GB RAM or more", 16),
        ("16+ GB RAM", 16),
    ],
)
def test_ram_patterns(text, ram):
    result = intent(f"laptop with {text}")
    assert result.min_ram_gb == ram
    assert not result.needs_clarification


@pytest.mark.parametrize(
    ("text", "kg"),
    [
        ("at most 1.5 kg", 1.5),
        ("under 1.5 kg", 1.5),
        ("below 2 kilograms", 2.0),
        ("max 1.3kg", 1.3),
        ("lighter than 1.4 kg", 1.4),
        ("weighing less than 1.2 kg", 1.2),
        ("weight at most 1.5 kg", 1.5),
        ("1.5 kg or less", 1.5),
        ("no more than 1.5 kg", 1.5),
        ("not more than 1.5 kg", 1.5),
    ],
)
def test_weight_patterns(text, kg):
    result = intent(f"laptop {text}")
    assert result.max_weight_kg == kg
    assert not result.needs_clarification


def test_reference_request_from_the_guide():
    result = intent(
        "I need a laptop for Python development and travel, under $900, "
        "with at least 16 GB RAM and weight at most 1.5 kg."
    )
    assert result.max_price_cents == 89999
    assert result.min_ram_gb == 16
    assert result.max_weight_kg == 1.5
    assert result.semantic_query == "laptop for Python development and travel"
    assert result.explicit_fields == ["max_price_cents", "max_weight_kg", "min_ram_gb"]
    assert result.constraints().require_in_stock is True


def test_in_stock_phrase_is_explicit():
    result = intent("laptop for study that is in stock")
    assert result.require_in_stock is True
    assert "require_in_stock" in result.explicit_fields


def test_semantic_query_without_constraints_is_preserved():
    result = intent("Portable laptop for coding and commuting")
    assert result.semantic_query == "Portable laptop for coding and commuting"
    assert result.explicit_fields == []


def test_constraint_only_first_turn_uses_default_semantic():
    assert intent("under $700").semantic_query == "laptop"


# --- ambiguity, contradiction, unsupported -------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "something cheap",
        "an affordable laptop",
        "cheapest laptop for study",
        "laptop for $900",
        "I have $900",
        "laptop under $-5",
        "laptop under $0",
        "laptop under $899.999",
        "laptop under €900",
        "laptop under 900 euros",
        "laptop over $500",
        "laptop at least $500",
        "laptop between $500 and $900",
        "laptop under $500 and at most $900",
        "laptop with at most 8 GB RAM",
        "laptop with at least 16.5 GB RAM",
        "laptop at least 1 kg",
        "laptop under 3 lbs",
        "laptop under 3 pounds",
        "laptop at most 0 kg",
    ],
)
def test_ambiguous_or_unsupported_asks_a_question(text):
    result = intent(text)
    assert result.needs_clarification
    assert result.clarification_question


def test_cheap_with_budget_is_a_soft_preference():
    result = intent("cheap laptop under $500")
    assert not result.needs_clarification
    assert result.max_price_cents == 49999


def test_bare_ram_without_direction_is_not_a_hard_constraint():
    result = intent("16 GB RAM laptop for coding")
    assert result.min_ram_gb is None
    assert not result.needs_clarification
    assert "16 GB RAM" in result.semantic_query


def test_contradictory_values_ask_which_one():
    result = intent("laptop under $500, at most $900")
    assert result.needs_clarification
    assert "conflicting" in result.clarification_question


def test_cheap_never_invents_a_budget():
    result = intent("Something cheap.")
    assert result.max_price_cents is None
    assert result.needs_clarification


# --- structured controls -------------------------------------------------------------


def test_controls_are_authoritative_and_merge_with_text():
    result = intent("laptop for coding", controls=Filters(max_price_cents=90000, min_ram_gb=16))
    assert (result.max_price_cents, result.min_ram_gb) == (90000, 16)
    assert not result.needs_clarification
    assert "max_price_cents" in result.explicit_fields


def test_same_value_in_text_and_control_is_not_a_conflict():
    result = intent("laptop at most $900", controls=Filters(max_price_cents=90000))
    assert not result.needs_clarification


def test_conflicting_text_and_control_ask_and_show_both_values():
    result = intent("laptop under $700", controls=Filters(max_price_cents=90000))
    assert result.needs_clarification
    assert "69999" in result.clarification_question
    assert "90000" in result.clarification_question


def test_control_conflicts_with_clear_request():
    result = intent("forget the weight limit", controls=Filters(max_weight_kg=1.5))
    assert result.needs_clarification


def test_stock_control_conflict():
    result = intent("laptop in stock", controls=Filters(require_in_stock=False))
    assert result.needs_clarification


# --- session follow-ups --------------------------------------------------------------


def first_turn():
    return parse_and_merge(
        "laptop for coding and commuting, under $900, at least 16 GB RAM, at most 1.5 kg"
    )


def test_followup_replaces_budget_and_preserves_other_filters():
    state = first_turn().session
    result = parse_and_merge("under $700 instead", session=state)
    assert result.intent.max_price_cents == 69999
    assert result.intent.min_ram_gb == 16
    assert result.intent.max_weight_kg == 1.5
    assert result.intent.semantic_query == "laptop for coding and commuting"
    assert result.intent.explicit_fields == ["max_price_cents"]


def test_followup_forget_weight_limit_clears_only_that_field():
    state = first_turn().session
    result = parse_and_merge("Forget the weight limit", session=state)
    assert result.intent.max_weight_kg is None
    assert result.intent.max_price_cents == 89999
    assert result.intent.min_ram_gb == 16
    assert result.session.max_weight_kg is None


@pytest.mark.parametrize(
    "phrase", ["remove the budget", "drop the budget limit", "no budget limit"]
)
def test_other_clear_phrases(phrase):
    state = first_turn().session
    assert parse_and_merge(phrase, session=state).intent.max_price_cents is None


def test_followup_with_new_semantic_text_replaces_semantic_intent():
    state = first_turn().session
    result = parse_and_merge("something lighter for travel", session=state)
    assert result.intent.semantic_query == "something lighter for travel"
    assert result.intent.max_price_cents == 89999


def test_clarification_does_not_change_the_session():
    state = first_turn().session
    result = parse_and_merge("laptop for $900", session=state)
    assert result.intent.needs_clarification
    assert result.session == state


def test_reset_session_inherits_nothing():
    first_turn()
    result = parse_and_merge("laptop for study", session=SessionState())
    assert result.intent.max_price_cents is None
    assert result.intent.min_ram_gb is None
    assert result.intent.max_weight_kg is None


def test_sessions_are_isolated_values_not_shared_state():
    a = first_turn().session
    b = SessionState()
    parse_and_merge("under $100", session=a)
    assert b == SessionState()
    assert a.max_price_cents == 89999


def test_history_is_bounded():
    state = SessionState()
    for i in range(12):
        state = parse_and_merge(f"laptop number {i}", session=state).session
    assert len(state.history) == 5
    assert state.history[-1] == "laptop number 11"


# --- normalization and bounds --------------------------------------------------------


def test_unicode_and_whitespace_are_normalized():
    assert normalize_query("  laptop　　for\tcoding \n") == "laptop for coding"
    assert intent("laptop under ＄９００").max_price_cents == 89999


def test_control_characters_are_removed():
    assert normalize_query("lap\x00top\x07 now") == "lap top now"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_query_is_rejected(bad):
    with pytest.raises(InvalidQueryError):
        normalize_query(bad)


def test_query_length_is_bounded():
    assert normalize_query("a" * 2000)
    with pytest.raises(InvalidQueryError, match="2000"):
        normalize_query("a" * 2001)
    with pytest.raises(InvalidQueryError):
        parse_and_merge("a" * 50, max_chars=10)


def test_negation_and_units_are_preserved_in_semantic_text():
    result = intent("laptop for coding, not gaming")
    assert result.semantic_query == "laptop for coding, not gaming"
