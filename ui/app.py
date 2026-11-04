"""Streamlit demo UI. It talks to the API only, through recommender.client.

Run the API first:   python -m uvicorn recommender.api:app --host 127.0.0.1 --port 8000
Then the UI:         python -m streamlit run ui/app.py
Set RECOMMENDER_API_URL to point the UI at a different API address.
"""

from __future__ import annotations

import os

import streamlit as st

from recommender import client as rc
from recommender.schemas import RecommendationRequest, RecommendationResponse

API_URL = os.environ.get("RECOMMENDER_API_URL", rc.DEFAULT_BASE_URL)
START_HINT = "Start it with: python -m uvicorn recommender.api:app --host 127.0.0.1 --port 8000"
EXAMPLE = "Portable laptop for coding and commuting"

st.set_page_config(page_title="Product Recommendation Agent", layout="wide")


def reset() -> None:
    """Clear every active constraint, the conversation and any shown result."""
    for key in ("budget", "min_ram", "max_weight"):
        st.session_state[key] = None
    st.session_state["in_stock"] = True
    st.session_state["query"] = ""
    st.session_state["session"] = None
    st.session_state["response"] = None
    st.session_state["error"] = None


for key, default in (("session", None), ("response", None), ("error", None)):
    st.session_state.setdefault(key, default)

st.title("RAG-Based Intelligent Product Recommendation Agent")
st.info(
    "Synthetic demo catalog / snapshot prices. These are fictional laptops: not live stock, "
    "not real retailers, and the catalog contains no performance benchmarks."
)

with st.sidebar:
    st.header("Filters (optional)")
    st.number_input(
        "Maximum budget (USD)", min_value=0.0, value=None, step=50.0, format="%.2f", key="budget"
    )
    st.number_input("Minimum RAM (GB)", min_value=1, value=None, step=1, key="min_ram")
    st.number_input(
        "Maximum weight (kg)", min_value=0.1, value=None, step=0.1, format="%.1f", key="max_weight"
    )
    st.toggle("In stock only", value=True, key="in_stock")
    st.button("Reset conversation", on_click=reset)
    st.caption(
        "You can also type limits in your request, and follow up with things like "
        "'under $700 instead'. If a control here disagrees with your text, you will be asked "
        "which one to use."
    )
    status = rc.make_client(API_URL).ready()
    if status is None:
        st.warning("Service: not reachable or not ready.")
    else:
        st.caption(f"Service: {status['status']} - {status['catalog_size']} products")

with st.form("ask"):
    st.text_input("What are you looking for?", key="query", placeholder=EXAMPLE)
    submitted = st.form_submit_button("Get recommendations")


def submit() -> None:
    query = (st.session_state.get("query") or "").strip()
    if not query:
        st.session_state.error = ("invalid_request", "Please describe what you need first.")
        return
    try:
        filters = rc.build_filters(
            st.session_state.get("budget"),
            st.session_state.get("min_ram"),
            st.session_state.get("max_weight"),
            st.session_state.get("in_stock", True),
        )
        request = RecommendationRequest(
            query=query, filters=filters, session=st.session_state.session
        )
    except ValueError as exc:
        st.session_state.error = ("invalid_request", str(exc))
        return
    with st.spinner("Searching the catalog..."):
        try:
            response = rc.make_client(API_URL).recommend(request)
        except rc.ClientError as exc:
            st.session_state.error = (exc.kind, exc.message)
            return
    st.session_state.error = None
    st.session_state.response = response
    st.session_state.session = response.session  # held here; the server keeps no state


def show_attribute(label: str, value, unit: str = "") -> str:
    return f"{label}: not provided by this catalog" if value is None else f"{label}: {value}{unit}"


def render_item(item) -> None:
    with st.container(border=True):
        st.subheader(item.title)
        stock = "In stock" if item.in_stock else "Out of stock"
        st.caption(
            f"{item.product_id} | {rc.format_money(item.price_cents, item.currency)} | "
            f"{stock} (snapshot)"
        )
        attrs = item.attributes
        st.text(
            " | ".join(
                [
                    show_attribute("RAM", attrs["ram_gb"], " GB"),
                    show_attribute("Storage", attrs["storage_gb"], " GB"),
                    show_attribute("Weight", attrs["weight_kg"], " kg"),
                    show_attribute("GPU", attrs["gpu_type"]),
                ]
            )
        )
        for reason in item.reasons:
            st.text(f"- {reason}")
        with st.expander("Evidence and sources"):
            st.text("Evidence fields: " + (", ".join(item.evidence_refs) or "none"))
            if item.unknown_fields:
                st.text("Unknown in this catalog: " + ", ".join(item.unknown_fields))
            st.text(f"Sample source identifier: {item.source_url}")
            st.caption("A sample identifier from a synthetic catalog, not a purchase link.")


def render_response(response: RecommendationResponse) -> None:
    filters = response.interpreted_filters.model_dump()
    st.markdown("**Active filters:** " + " | ".join(rc.describe_filters(filters)))
    st.caption(f"Request topic: {filters['semantic_query']}")

    if response.status == "needs_clarification":
        st.info(response.clarification_question)
    elif response.status == "no_match":
        st.warning("No catalog product meets all of your constraints.")
        for line in response.limitations:
            st.text(f"- {line}")
        if response.clarification_question:
            st.info(response.clarification_question)
    else:
        if response.explanation_mode == "llm":
            st.caption("Explanation chosen by a language model and validated against the catalog.")
        else:
            st.caption("Basic explanation: verified catalog facts rendered with fixed templates.")
        if response.fallback_reason and response.fallback_reason != "llm_disabled":
            st.caption(
                f"(The language model was unavailable or rejected: {response.fallback_reason})"
            )
        for item in response.recommendations:
            render_item(item)
        with st.expander("Limitations and notes"):
            for line in response.limitations:
                st.text(f"- {line}")


if submitted:
    submit()

if st.session_state.error:
    kind, message = st.session_state.error
    st.error(message + (f"\n\n{START_HINT}" if kind == "unavailable" else ""))
elif st.session_state.response is not None:
    render_response(st.session_state.response)
else:
    st.caption(f"Example: {EXAMPLE}")
