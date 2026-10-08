"""
agent.py -- the LangGraph agent: question -> choose tool -> run -> answer.

Graph:   START -> agent --(tool call?)--> tools --(ok)--> END
                   |                        |
                   +--(no tool: reply)--> END   (error) -> agent (retry)
  agent  the local model (via Ollama) reads the question + chat history and
         either calls one of the controlled tools or replies directly (to ask
         a clarifying question or say what it can help with).
  tools  runs the chosen tool. The answer text is built by the tool from the
         real rows, so names and numbers always match the database. Bad inputs
         (validation errors) go back to the model so it can fix them.

The model never sees or writes SQL -- only tool names and parameters -- and
never writes facts: it only chooses which tool to call and with what filters.
"""

import json
import re
import uuid
from datetime import date

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

MODEL_NAME = "qwen3:1.7b"
MAX_HISTORY_TURNS = 6    # earlier turns kept for follow-up questions
RECURSION_LIMIT = 10     # max graph steps per question (~4 tool rounds)

SYSTEM_PROMPT = """You are the Candidate Assistant for an HR team. You answer questions about job
applicants and about the email-processing pipeline, using ONLY the provided tools.
Today's date is {today}.

Data you can reach through tools:
- candidates: one row per applicant (name, email, phone, city, years of experience,
  education, latest company, skills, CV file).
- messages: one row per application email the pipeline processed (data extracted, skipped
  because not a CV, or failed, with the reason and date).

Rules:
1. For ANY question about candidates, counts, emails or statistics, call a tool.
   The tool's result is shown to the user as the answer.
2. Pick the single best tool:
   - how many / show / list / who / find candidates -> search_candidates
   - breakdown "by city / degree / experience / company" -> group_candidates
   - one specific person -> get_candidate
   - email totals (processed = ALL emails handled, whatever the result;
     extracted / skipped / failed are parts of it) -> email_stats
   - why emails failed or were skipped -> failure_reasons
   - emails per day/week/month -> processing_timeline
   - candidates who sent their CV more than once / repeat or duplicate CVs -> repeat_applicants
3. Only pass the filters the user actually asked for; leave every other filter out.
   "How many candidates in total" -> search_candidates with no filters at all.
   If a tool returns an error, call it again with fixed parameters.
   Experience is in whole years. "3+ years" means min_years=3.
   "MBA", "MS", "MSc" mean min_degree="masters"; "BS", "BBA", "graduate" mean "bachelors".
4. Follow-up questions ("now only Lahore", "what about 5+ years", "only those with a
   masters") refine the previous call: use the SAME tool, keep ALL its previous filters,
   and add or change only the new one. Example: earlier search_candidates(min_years=3),
   new question "only those with a masters" -> search_candidates(min_years=3, min_degree="masters").
   A question that does NOT refer back (no "now", "only those", "what about", "them")
   is a new question: use only the filters it states itself.
5. Dates: convert "this week", "last month", "today" to YYYY-MM-DD using today's date.
6. If a question is unclear, ask one short clarifying question instead of guessing.
   If it is unrelated to candidates or the pipeline, say what you can help with.
"""


def _llm():
    return ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192, reasoning=False)


def build_graph():
    llm = _llm().bind_tools(TOOLS)

    def agent(state: MessagesState, config: RunnableConfig):
        # /no_think: Qwen3's own switch to skip thinking (slow on CPU)
        system = SystemMessage(SYSTEM_PROMPT.format(today=date.today().isoformat()) + "\n/no_think")
        msg = _text_to_tool_call(llm.invoke([system] + state["messages"]))
        previous = config.get("configurable", {}).get("previous_calls") or []
        return {"messages": [_keep_previous_filters(msg, previous)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_conditional_edges("tools", _after_tools, ["agent", END])
    return graph.compile()


TOOL_NAMES = {t.name for t in TOOLS}

# wording that refines the previous question ("now only Lahore", "what about 5+ years")
FOLLOW_UP = re.compile(
    # only wording that clearly points back to the previous results; anything
    # else (even "only ..." or "and ...") is a new question with fresh filters
    r"^\s*(now|what about|how about)\b"
    r"|\b(those|them|these|they|their|the same|previous|above|that list|from that)\b", re.I)


def _text_to_tool_call(msg):
    """Small models sometimes write the tool call as plain text, e.g.
    {"name": "email_stats", "arguments": {...}}, instead of making a real
    call. Turn that text into a real tool call so it still runs."""
    if msg.tool_calls:
        return msg
    text = _clean(msg.content)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return msg
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return msg
    if not isinstance(data, dict) or data.get("name") not in TOOL_NAMES:
        return msg
    args = data.get("arguments", data.get("parameters", {}))
    if not isinstance(args, dict):
        return msg
    return AIMessage("", tool_calls=[{"name": data["name"], "args": args, "id": f"call_{uuid.uuid4().hex[:12]}"}])


def _keep_previous_filters(msg, previous):
    """On a follow-up ("now only those with a masters"), a call to the same
    tool as last time keeps the previous filters; the new ones override them.
    Done in code because the small model often drops the old filters."""
    if not previous or not msg.tool_calls:
        return msg
    last = {c["name"]: c["args"] for c in previous}
    calls = [{**c, "args": {**last[c["name"]], **c["args"]}} if c["name"] in last else c
             for c in msg.tool_calls]
    return AIMessage(msg.content, tool_calls=calls)


def _after_tools(state: MessagesState):
    """Done once the tools succeeded (they wrote the answer); back to the
    model only if a call failed, so it can retry with fixed parameters."""
    for m in reversed(state["messages"]):
        if not isinstance(m, ToolMessage):
            break
        if getattr(m, "status", None) == "error":
            return "agent"
    return END


def _clean(text):
    # in case the model still emits thinking; the opening <think> tag is
    # sometimes missing, so drop everything up to the last </think>
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.rsplit("</think>", 1)[-1].strip()


def ask(graph, question, history=()):
    """history: earlier turns as dicts {"role": "user"|"assistant", "content",
    "calls", "tool_calls"}. Returns {"answer", "artifacts", "calls", "tool_calls", "errors"}."""
    # Earlier turns go in as a short note of past questions and the tool calls
    # they used (so follow-ups can refine them), not as past answers: given
    # answers, a small model copies their style and skips calling a tool.
    earlier, last_q = [], None
    for turn in list(history)[-2 * MAX_HISTORY_TURNS:]:
        if turn["role"] == "user":
            last_q = turn["content"]
        elif last_q is not None:
            used = "; ".join(turn.get("calls") or []) or "no tool"
            earlier.append(f'- "{last_q}" -> {used}')
            last_q = None
    # Only pass earlier calls when the question refers back to them; otherwise
    # the small model copies old filters into new questions ("show all" -> 3+ years).
    follow_up = bool(earlier) and bool(FOLLOW_UP.search(question))
    previous = []
    if follow_up:
        previous = next((t.get("tool_calls") or [] for t in reversed(list(history))
                         if t["role"] == "assistant"), [])
    if follow_up:
        question_text =("Earlier questions in this chat and the tool calls used:\n" + "\n".join(earlier)
                         + f"\n\nNew question (call a tool to answer it): {question}")
    else:
        question_text = question
    messages = [HumanMessage(question_text)]
    start = len(messages)

    try:
        result = graph.invoke({"messages": messages}, config={
            "recursion_limit": RECURSION_LIMIT, "configurable": {"previous_calls": previous}})
    except GraphRecursionError:
        return {"answer": "Sorry, I couldn't work that out. Could you rephrase the question "
                          "or make it more specific?", "artifacts": [], "calls": [], "tool_calls": [],
                "errors": []}

    new = result["messages"][start:]
    artifacts, calls, tool_calls, errors = [], [], [], []
    requested = {c["id"]: c for m in new if isinstance(m, AIMessage) for c in m.tool_calls}
    for m in new:
        if isinstance(m, ToolMessage):
            if getattr(m, "status", None) == "error" or not m.artifact:
                errors.append(m.content)
            else:
                artifacts.append(m.artifact)
                calls.append(m.artifact["call"])
                if m.tool_call_id in requested:
                    c = requested[m.tool_call_id]
                    tool_calls.append({"name": c["name"], "args": c["args"]})
    if artifacts:
        answer = "\n\n".join(a["answer"] for a in artifacts)
    elif errors:
        # the model's text after a failed call is about fixing the call, not an answer
        answer = ("Sorry, I couldn't work that out. Could you rephrase the question, e.g. "
                  "\"Show candidates in Lahore with 3+ years of experience\"?")
    else:
        answer = _clean(new[-1].content if new else "") or "Sorry, I don't have an answer for that."
    return {"answer": answer, "artifacts": artifacts, "calls": calls, "tool_calls": tool_calls,
            "errors": errors}


if __name__ == "__main__":
    import sys
    g = build_graph()
    out = ask(g, " ".join(sys.argv[1:]) or "How many emails have been processed?")
    print(out["answer"])
    print("calls:", out["calls"], "| errors:", out["errors"])
