"""
app.py -- Streamlit UI for the Candidate Assistant.

Run:   streamlit run app.py

Layout:
  sidebar  live stats cards (emails processed / extracted / failed / candidates),
           example questions, clear-chat button
  main     chat history; each assistant reply shows the short answer plus the
           full result (table / chart / numbers), a CSV download and CV links

The UI only talks to agent.ask(); it never touches SQL itself, except the
sidebar stats, which call the same read-only email_stats tool directly.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

import agent
import db
import tools

APP_DIR = Path(__file__).resolve().parent
MAX_CV_LINKS = 10


EXAMPLES = [
    "How many emails have been processed?",
    "Show candidates with 3+ years of experience",
    "Who has a masters degree?",
    "Why did emails fail?",
    "Emails processed per day",
    "Breakdown of candidates by education",
]

st.set_page_config(page_title="Candidate Assistant", page_icon="🧑‍💼", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 2rem; max-width: 1200px; }
  h1 { font-size: 1.8rem !important; margin-bottom: 0 !important; }
  .sub { color: #5b6475; margin-bottom: 1.2rem; }
  .card { background: #ffffff; border: 1px solid #dde1e8; border-radius: 12px;
          padding: 10px 14px; margin-bottom: 8px; }
  .card .label { color: #5b6475; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .04em; }
  .card .value { font-size: 1.6rem; font-weight: 600; color: #1d2433; }
  .card.ok .value { color: #13866f; }
  .card.bad .value { color: #c2581d; }
  .call { font-family: Consolas, monospace; font-size: 0.75rem; color: #7a4cc2; }
  @media (prefers-color-scheme: dark) {
    .card { background: #1c2230; border-color: #2d3546; }
    .card .value { color: #e8ebf2; }
    .sub, .card .label { color: #9aa3b5; }
  }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading the agent...")
def get_graph():
    return agent.build_graph()


@st.cache_data(ttl=60)
def sidebar_stats():
    _, artifact = tools.email_stats.func()
    return artifact["stats"]


def card(label, value, tone=""):
    st.markdown(f'<div class="card {tone}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


def cv_path(cv_file):
    if not cv_file:
        return None
    path = (APP_DIR / cv_file).resolve()
    # only serve files inside the app folder
    return path if path.is_file() and APP_DIR in path.parents else None


def render_cv_links(rows, key):
    with_cv = [r for r in rows if r.get("CV")][:MAX_CV_LINKS]
    if not with_cv:
        return
    missing = 0
    cols = st.columns(min(len(with_cv), 3))
    for i, r in enumerate(with_cv):
        path = cv_path(r["CV"])
        if path is None:
            missing += 1
            continue
        cols[i % len(cols)].download_button(
            f"📄 {r['Name'] or path.name}", path.read_bytes(), file_name=path.name,
            mime="application/pdf" if path.suffix.lower() == ".pdf" else None,
            key=f"{key}-cv-{i}", width="stretch")
    if missing:
        st.caption(f"{missing} CV file(s) not found. Copy the pipeline's CVs\\ folder "
                   f"next to app.py to enable CV links.")


def render_artifact(a, key):
    kind, rows = a["kind"], a["rows"]
    st.markdown(f"**{a['title']}** &nbsp; <span class='call'>{a['call']}</span>", unsafe_allow_html=True)

    if kind == "number":
        st.metric("Count", a["total"])
        return

    if kind == "stats":
        s = a["stats"]
        c = st.columns(4)
        c[0].metric("Emails processed", s["total_emails_processed"])
        c[1].metric("Data extracted", s["data_extracted"])
        c[2].metric("Skipped (not CV)", s["skipped_not_cv"])
        c[3].metric("Failed", s["failed"])

    if not rows:
        st.info("No matching results.")
        return

    df = pd.DataFrame(rows)
    if kind == "groups":
        label, value = df.columns[0], df.columns[1]
        st.bar_chart(df.set_index(label)[value], horizontal=True)
    elif kind == "timeline":
        st.bar_chart(df.set_index("Period")[["Extracted", "Skipped", "Failed"]],
                     color=["#13866f", "#8a93a6", "#c2581d"])

    if kind == "table":
        df = df.drop(columns=["CV"], errors="ignore")
    st.dataframe(df, width="stretch", hide_index=True)
    if kind == "table" and a["total"] > len(rows):
        st.caption(f"Showing {len(rows)} of {a['total']} matches. Add filters to narrow it down.")
    st.download_button("⬇ Download CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                       file_name="results.csv", mime="text/csv", key=f"{key}-csv")
    if kind == "table":
        render_cv_links(rows, key)


def render_turn(turn, idx):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for j, a in enumerate(turn.get("artifacts") or []):
            # just the answer by default; the table/chart is one click away
            with st.expander("Show details"):
                render_artifact(a, key=f"t{idx}-a{j}")
        if turn.get("errors"):
            with st.expander("Details"):
                for e in turn["errors"]:
                    st.caption(e)


# ------------------------------------------------------------------ sidebar
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Pipeline at a glance")
    try:
        s = sidebar_stats()
        card("Emails processed", s["total_emails_processed"])
        card("Data extracted", s["data_extracted"], "ok")
        card("Emails failed", s["failed"], "bad")
        card("Skipped (not a CV)", s["skipped_not_cv"])
        card("Total candidates", s["total_candidates (all time)"])
    except FileNotFoundError as e:
        st.error(str(e))

    st.subheader("Try asking")
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            st.session_state.pending = q

    st.divider()
    if st.button("🗑 Clear chat", width="stretch"):
        st.session_state.history = []
        st.rerun()
    st.caption(f"Model: {agent.MODEL_NAME} (local, via Ollama) · data: {db.DB_PATH.name} (read-only)")


# --------------------------------------------------------------------- main
st.title("Candidate Assistant")
st.markdown('<div class="sub">Ask about applicants and the CV email pipeline in plain language.</div>',
            unsafe_allow_html=True)

for i, turn in enumerate(st.session_state.history):
    render_turn(turn, i)

question = st.chat_input("e.g. graphic designers in Lahore with 3+ years") or st.session_state.pop("pending", None)

if question:
    history = st.session_state.history
    history.append({"role": "user", "content": question})
    render_turn(history[-1], len(history) - 1)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (the local model can take a little while)"):
            try:
                out = agent.ask(get_graph(), question, history[:-1])
            except Exception as e:  # Ollama not running, model missing, etc.
                out = {"answer": f"Sorry, something went wrong talking to the model: `{e}`. "
                                 f"Is Ollama running (`ollama serve`) with `{agent.MODEL_NAME}` pulled?",
                       "artifacts": [], "calls": [], "errors": []}
    history.append({"role": "assistant", "content": out["answer"], "artifacts": out["artifacts"],
                    "calls": out["calls"], "tool_calls": out["tool_calls"], "errors": out["errors"]})
    st.rerun()
