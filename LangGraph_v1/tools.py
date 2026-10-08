"""
tools.py -- the controlled SQL tools the agent can call.

The model never writes SQL. It can only call these functions; each one runs
a fixed, pre-written SELECT on a read-only connection, with the model's
values bound as ? parameters (never pasted into the SQL text) and validated
first.

Every tool returns (answer, artifact):
  answer   -- the reply the user reads, built here from the real rows (the
              model never rewrites it, so names and numbers stay exact)
  artifact -- the full result for the UI: {"kind", "title", "answer", "rows", "total", "call"}
"""

import re
from contextlib import closing
from collections import Counter
from datetime import datetime
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

import db

MAX_ROWS = 50          # rows sent to the UI table per answer
MAX_ROWS_FOR_LLM = 10  # rows listed in the answer text (all rows are in the details table)


class ToolInputError(ValueError):
    pass


# ---------------------------------------------------------------- validation
def _text(value, field, max_len=60):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if len(value) > max_len:
        raise ToolInputError(f"{field} is too long (max {max_len} characters)")
    return value


def _years(value, field):
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{field} must be a whole number of years")
    if not 0 <= value <= 60:
        raise ToolInputError(f"{field} must be between 0 and 60")
    return value


def _date(value, field):
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ToolInputError(f"{field} must be a date like 2026-10-07")


def _call(tool_name, /, **kwargs):
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items() if v not in (None, [], ""))
    return f"{tool_name}({args})"


# ------------------------------------------------------------ shared filters
DegreeName = Literal["matric", "intermediate", "bachelors", "masters", "phd"]


class CandidateFilters(BaseModel):
    skills: Optional[list[str]] = Field(
        None, description='Skills the candidate must ALL have, e.g. ["Photoshop", "Illustrator"]. '
                          'Matched inside the skills text.')
    city: Optional[str] = Field(None, description='City the candidate lives in, e.g. "Lahore".')
    min_years: Optional[int] = Field(
        None, description="Minimum whole years of experience (e.g. 3 for '3+ years'). "
                          "Candidates with unknown experience are excluded when this is set.")
    max_years: Optional[int] = Field(None, description="Maximum whole years of experience.")
    min_degree: Optional[DegreeName] = Field(
        None, description="Minimum education level. Order: matric < intermediate < bachelors < masters < phd. "
                          "An MBA/MS/MSc counts as masters; BS/BBA/BSc as bachelors.")
    company: Optional[str] = Field(None, description="Text inside the candidate's latest company name.")
    name: Optional[str] = Field(None, description="Text inside the candidate's name.")

    @model_validator(mode="before")
    @classmethod
    def _drop_placeholders(cls, data):
        """Small models often fill every filter with placeholders ("All", 0,
        "skills": "All") when they mean "no filter". Treat those as unset."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for field in _FILTER_FIELDS:
            v = data.get(field)
            if isinstance(v, str) and v.strip().lower() in _PLACEHOLDERS:
                data[field] = None
        skills = data.get("skills")
        if isinstance(skills, str):
            skills = [s for s in skills.split(",")]
        if isinstance(skills, list):
            skills = [s for s in skills if str(s).strip().lower() not in _PLACEHOLDERS]
            data["skills"] = skills or None
        if data.get("min_years") == 0:
            data["min_years"] = None  # "0+ years" is no filter
            if data.get("max_years") == 0:
                data["max_years"] = None
        return data


_FILTER_FIELDS = ("skills", "city", "min_years", "max_years", "min_degree", "company", "name")
_PLACEHOLDERS = {"", "all", "any", "none", "null", "n/a", "na", "*", "everyone", "anywhere", "unknown"}


# a "skill" that is really an experience or degree filter ("3+ years", "masters")
_NOT_A_SKILL = re.compile(r"\b(years?|yrs?|experience|degree)\b|"
                          r"^(masters?|bachelors?|phd|mba|ms|msc|bs|bsc|bba|matric|intermediate|"
                          r"cvs?|resumes?|candidates?|applicants?)$", re.I)


def _where(f):
    """WHERE clause + parameters for the candidate filters (validated)."""
    where, params = [], []
    for s in f.get("skills") or []:
        s = _text(s, "skill")
        # small models sometimes put "3+ years" or "masters" in skills; those
        # have their own filters, so ignore them here
        if s and not _NOT_A_SKILL.search(s):
            where.append("skills LIKE ?")
            params.append(f"%{s}%")
    for field, column in (("city", "city"), ("company", "past_companies"), ("name", "full_name")):
        v = _text(f.get(field), field)
        if v:
            where.append(f"{column} LIKE ?")
            params.append(f"%{v}%")
    lo, hi = _years(f.get("min_years"), "min_years"), _years(f.get("max_years"), "max_years")
    if lo is not None and hi is not None and lo > hi:
        raise ToolInputError("min_years cannot be greater than max_years")
    if lo is not None:
        where.append("years(experience) >= ?")
        params.append(lo)
    if hi is not None:
        where.append("years(experience) <= ?")
        params.append(hi)
    if f.get("min_degree"):
        level = db.DEGREE_LEVELS.get(str(f["min_degree"]).lower())
        if level is None:
            raise ToolInputError("min_degree must be one of: matric, intermediate, bachelors, masters, phd")
        where.append("degree_level(education) >= ?")
        params.append(level)
    return (" WHERE " + " AND ".join(where)) if where else "", params


def _candidate_rows(conn, where, params, limit):
    email = db.candidate_email_column(conn)
    sql = (f"SELECT full_name, {email} AS email, phone, city, experience, years(experience) AS years, "
           f"education, past_companies, skills, cv_file, received_at FROM candidates{where} "
           f"ORDER BY years(experience) IS NULL, years(experience) DESC, full_name LIMIT ?")
    return [
        {"Name": r["full_name"], "Email": r["email"], "Phone": r["phone"], "City": r["city"],
         "Years": r["years"], "Experience": r["experience"], "Education": r["education"],
         "Latest company": r["past_companies"], "Skills": r["skills"], "CV": r["cv_file"],
         "Applied": (r["received_at"] or "")[:10]}
        for r in conn.execute(sql, params + [limit])
    ]


# ------------------------------------------------- answers written in code
# The answer the user reads is built here from the real rows, not written by
# the model: a small model garbled names, years and degrees when it rewrote
# them. The model only chooses the tool and its filters.

def _skills(f):
    return [s.strip() for s in f.get("skills") or [] if s and s.strip() and not _NOT_A_SKILL.search(s)]


def _describe(f):
    """'with 3+ years of experience, in Lahore' from the filters used."""
    parts = []
    if _skills(f):
        parts.append("with skills " + ", ".join(_skills(f)))
    lo, hi = f.get("min_years"), f.get("max_years")
    if lo is not None and hi is not None:
        parts.append(f"with {lo}-{hi} years of experience")
    elif lo is not None:
        parts.append(f"with {lo}+ years of experience")
    elif hi is not None:
        parts.append(f"with at most {hi} years of experience")
    if f.get("min_degree"):
        parts.append(f"with at least a {f['min_degree']} degree")
    if f.get("city"):
        parts.append(f"in {f['city']}")
    if f.get("company"):
        parts.append(f"at {f['company']}")
    if f.get("name"):
        parts.append(f"named like '{f['name']}'")
    return (" " + ", ".join(parts)) if parts else ""


def _latest_company(r):
    return (r["Latest company"] or "").split(",")[0].strip()


def _one_line(r):
    """'**Huma Sadaf**: 9 years, Masters, latest company SPRINTX'"""
    details = [r["Experience"] or "experience unknown",
               db.DEGREE_NAMES[db.degree_level(r["Education"])]]
    if _latest_company(r):
        details.append(f"latest company {_latest_company(r)}")
    if r["City"]:
        details.append(r["City"])
    return f"**{r['Name']}**: " + ", ".join(details)


def _date_phrase(start_date, end_date):
    s, e = _date(start_date, "start_date"), _date(end_date, "end_date")
    if e and e >= datetime.now().strftime("%Y-%m-%d"):
        e = None  # "up to today" is the same as all time
    if s and e:
        return f" between {s} and {e}"
    if s:
        return f" since {s}"
    if e:
        return f" up to {e}"
    return ""


def _bullets(lines):
    return "\n".join(f"- {line}" for line in lines)


# ------------------------------------------------------- candidates tools
@tool(args_schema=CandidateFilters, response_format="content_and_artifact")
def search_candidates(**filters):
    """List candidates (names and details) matching filters (skills, city,
    years of experience, education level, latest company, name). Use for
    'how many', 'show', 'list', 'who', 'find' and shortlisting questions like
    'graphic designers in Lahore with 3+ years'."""
    with closing(db.connect()) as conn:
        where, params = _where(filters)
        total = conn.execute(f"SELECT COUNT(*) FROM candidates{where}", params).fetchone()[0]
        rows = _candidate_rows(conn, where, params, MAX_ROWS)
    desc = _describe(filters)
    if not total:
        answer = f"No candidates found{desc}."
    else:
        answer = f"Found **{total}** candidate(s){desc}:\n\n" + _bullets(_one_line(r) for r in rows[:MAX_ROWS_FOR_LLM])
        if total > MAX_ROWS_FOR_LLM:
            answer += f"\n\n...and {total - MAX_ROWS_FOR_LLM} more (see details)."
    return answer, {"kind": "table", "title": f"{total} matching candidate(s)", "answer": answer,
                    "rows": rows, "total": total, "call": _call("search_candidates", **filters)}


class GroupInput(CandidateFilters):
    by: Literal["city", "degree", "years", "company"] = Field(
        description="What to group candidates by: city, degree (education level), "
                    "years (experience bands) or company (latest company).")
    top: Optional[int] = Field(15, description="How many groups to return (max 30).")


def _band(y):
    if y is None:
        return "Unknown"
    if y == 0:
        return "Less than 1 year"
    for lo, hi in ((1, 2), (3, 5), (6, 10)):
        if lo <= y <= hi:
            return f"{lo}-{hi} years"
    return "More than 10 years"


@tool(args_schema=GroupInput, response_format="content_and_artifact")
def group_candidates(by, top=15, **filters):
    """Count candidates per group (by city, degree level, experience band or
    latest company), optionally within filters. Use for 'which cities ...',
    'breakdown by education', 'distribution of experience'."""
    top = max(1, min(int(top or 15), 30))
    # a filter on the grouped field hides the other groups (the small model
    # adds min_degree to "breakdown by education"), so drop it
    filters.pop({"degree": "min_degree", "city": "city", "company": "company"}.get(by, ""), None)
    if by == "years":
        filters.pop("min_years", None)
        filters.pop("max_years", None)
    with closing(db.connect()) as conn:
        where, params = _where(filters)
        if by == "city":
            values = [(r[0] or "").strip().title() or "Unknown"
                      for r in conn.execute(f"SELECT city FROM candidates{where}", params)]
        elif by == "company":
            values = [(r[0] or "").strip() or "Unknown"
                      for r in conn.execute(f"SELECT past_companies FROM candidates{where}", params)]
        elif by == "degree":
            values = [db.DEGREE_NAMES[r[0]] for r in
                      conn.execute(f"SELECT degree_level(education) FROM candidates{where}", params)]
        else:
            values = [_band(r[0]) for r in conn.execute(f"SELECT years(experience) FROM candidates{where}", params)]
    counts = Counter(values).most_common(top)
    rows = [{"Group": g, "Candidates": n} for g, n in counts]
    answer = (f"**{len(values)}** candidate(s){_describe(filters)}, by {by}:\n\n"
              + _bullets(f"{g}: {n}" for g, n in counts))
    return answer, {
        "kind": "groups", "title": f"Candidates by {by}", "answer": answer, "rows": rows, "total": len(values),
        "call": _call("group_candidates", by=by, **filters)}


class LookupInput(BaseModel):
    query: str = Field(description="Part of the candidate's name or email address.")


@tool(args_schema=LookupInput, response_format="content_and_artifact")
def get_candidate(query):
    """Look up specific candidates by (part of) their name or email address
    and return their full details. Use for 'show me <person>'."""
    q = _text(query, "query")
    if not q:
        raise ToolInputError("query must not be empty")
    with closing(db.connect()) as conn:
        email = db.candidate_email_column(conn)
        where = f" WHERE full_name LIKE ? OR {email} LIKE ?"
        rows = _candidate_rows(conn, where, [f"%{q}%", f"%{q}%"], 10)
    if not rows:
        answer = f"No candidate found matching '{q}'."
    else:
        answer = "\n\n".join(
            f"**{r['Name']}**\n" + _bullets([
                f"Experience: {r['Experience'] or 'unknown'}",
                f"Education: {r['Education'] or 'unknown'}",
                f"Latest company: {_latest_company(r) or 'unknown'}",
                f"City: {r['City'] or 'unknown'}",
                f"Email: {r['Email'] or '-'} · Phone: {r['Phone'] or '-'}",
                f"Skills: {r['Skills'] or '-'}",
            ]) for r in rows[:3])
        if len(rows) > 3:
            answer += f"\n\n...and {len(rows) - 3} more match(es) (see details)."
    return answer, {"kind": "table", "title": f"{len(rows)} candidate(s) found", "answer": answer,
                          "rows": rows, "total": len(rows), "call": _call("get_candidate", query=q)}


# ---------------------------------------------------------- messages tools
class DateRange(BaseModel):
    start_date: Optional[str] = Field(None, description="Only emails processed on/after this date, YYYY-MM-DD.")
    end_date: Optional[str] = Field(None, description="Only emails processed on/before this date, YYYY-MM-DD.")


def _date_where(start_date, end_date):
    s, e = _date(start_date, "start_date"), _date(end_date, "end_date")
    where, params = [], []
    if s:
        where.append("date(processed_at) >= ?")
        params.append(s)
    if e:
        where.append("date(processed_at) <= ?")
        params.append(e)
    return where, params


STATUS_LABELS = {"done": "Data extracted", "partial": "Data extracted (code-only)",
                 "not_cv": "Skipped (not a CV)", "failed": "Failed"}


@tool(args_schema=DateRange, response_format="content_and_artifact")
def email_stats(start_date=None, end_date=None):
    """Email processing totals from the pipeline: total emails processed
    (every email handled, whatever the result), how many had candidate data
    extracted, were skipped (not a CV) or failed, optionally within a date
    range; plus the total number of candidates. Use for 'how many emails
    were processed', 'how many failed this week', 'how many candidates'."""
    where, params = _date_where(start_date, end_date)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    with closing(db.connect()) as conn:
        by_status = dict(conn.execute(f"SELECT status, COUNT(*) FROM messages{w} GROUP BY status", params).fetchall())
        candidates = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    total = sum(by_status.values())
    stats = {
        "total_emails_processed": total,
        "data_extracted": by_status.get("done", 0) + by_status.get("partial", 0),
        "skipped_not_cv": by_status.get("not_cv", 0),
        "failed": by_status.get("failed", 0),
        "total_candidates (all time)": candidates,
    }
    rows = [{"Status": STATUS_LABELS.get(k, k), "Emails": v} for k, v in sorted(by_status.items())]
    answer = (f"**{total}** emails processed in total{_date_phrase(start_date, end_date)}: "
              f"{stats['data_extracted']} had candidate data extracted, {stats['failed']} failed and "
              f"{stats['skipped_not_cv']} were skipped (not a CV).\n\n"
              f"Total candidates so far: **{candidates}**.")
    return answer, {"kind": "stats", "title": "Email processing", "answer": answer, "rows": rows, "total": total,
                        "stats": stats, "call": _call("email_stats", start_date=start_date, end_date=end_date)}


class ReasonInput(DateRange):
    kind: Literal["failed", "skipped", "both"] = Field(
        "failed", description="failed = emails that failed; skipped = attachments that were not CVs; both.")


def _reason_label(reason):
    reason = reason or "unknown"
    if reason.startswith(("error:", "unexpected_error")):
        if re.search(r"Unable to find server|NameResolution|Max retries|Connection|timed out", reason, re.I):
            return "network error (no internet)"
        return "other error"
    return reason


@tool(args_schema=ReasonInput, response_format="content_and_artifact")
def failure_reasons(kind="failed", start_date=None, end_date=None):
    """Breakdown of WHY emails failed or were skipped (e.g. no_email_found,
    encrypted_pdf, network error), optionally within a date range."""
    where, params = _date_where(start_date, end_date)
    statuses = {"failed": ["failed"], "skipped": ["not_cv"], "both": ["failed", "not_cv"]}[kind]
    where.append(f"status IN ({','.join('?' * len(statuses))})")
    params += statuses
    with closing(db.connect()) as conn:
        reasons = [_reason_label(r[0]) for r in
                   conn.execute(f"SELECT reason FROM messages WHERE {' AND '.join(where)}", params)]
    counts = Counter(reasons).most_common()
    rows = [{"Reason": r, "Emails": n} for r, n in counts]
    what = {"failed": "failed", "skipped": "were skipped (not a CV)", "both": "failed or were skipped"}[kind]
    answer = f"**{len(reasons)}** emails {what}{_date_phrase(start_date, end_date)}."
    if counts:
        answer += " Reasons:\n\n" + _bullets(f"{r.replace('_', ' ')}: {n}" for r, n in counts)
    return answer, {
        "kind": "groups", "title": f"Reasons ({kind})", "answer": answer, "rows": rows, "total": len(reasons),
        "call": _call("failure_reasons", kind=kind, start_date=start_date, end_date=end_date)}


class TimelineInput(DateRange):
    period: Literal["day", "week", "month"] = Field("day", description="Group emails per day, week or month.")


@tool(args_schema=TimelineInput, response_format="content_and_artifact")
def processing_timeline(period="day", start_date=None, end_date=None):
    """How many emails were processed per day, week or month (data extracted,
    skipped, failed), optionally within a date range."""
    fmt = {"day": "%Y-%m-%d", "week": "%Y-W%W", "month": "%Y-%m"}[period]
    where, params = _date_where(start_date, end_date)
    where.append("processed_at IS NOT NULL")
    with closing(db.connect()) as conn:
        data = conn.execute(
            f"SELECT strftime('{fmt}', processed_at) AS p, status, COUNT(*) FROM messages "
            f"WHERE {' AND '.join(where)} GROUP BY p, status ORDER BY p", params).fetchall()
    table = {}
    for p, status, n in data:
        row = table.setdefault(p, {"Period": p, "Extracted": 0, "Skipped": 0, "Failed": 0})
        key = {"done": "Extracted", "partial": "Extracted", "not_cv": "Skipped"}.get(status, "Failed")
        row[key] += n
    rows = list(table.values())
    for r in rows:
        r["Total"] = r["Extracted"] + r["Skipped"] + r["Failed"]
    if not rows:
        answer = f"No emails processed{_date_phrase(start_date, end_date)}."
    else:
        answer = f"Emails processed per {period}{_date_phrase(start_date, end_date)}:\n\n" + _bullets(
            f"{r['Period']}: **{r['Total']}** ({r['Extracted']} extracted, {r['Failed']} failed, "
            f"{r['Skipped']} skipped)" for r in rows[-MAX_ROWS_FOR_LLM:])
        if len(rows) > MAX_ROWS_FOR_LLM:
            answer += f"\n\nShowing the last {MAX_ROWS_FOR_LLM} of {len(rows)} (see details)."
    return answer, {
        "kind": "timeline", "title": f"Emails per {period}", "answer": answer, "rows": rows,
        "total": sum(r["Total"] for r in rows),
        "call": _call("processing_timeline", period=period, start_date=start_date, end_date=end_date)}


class RepeatInput(DateRange):
    min_times: Optional[int] = Field(
        2, description="Minimum number of CVs sent by the same person (2 = 'more than once').")


# A repeat CV is a 'done' email whose candidate already existed:
#   updated              the new CV was newer and replaced the stored one
#   kept_newer_existing  the stored CV was newer, so it was kept
# (the first CV from a person has reason 'inserted'). People are matched by
# dedup_key, the same key the pipeline uses to merge candidates.
REPEAT_REASONS = ("updated", "kept_newer_existing")


@tool(args_schema=RepeatInput, response_format="content_and_artifact")
def repeat_applicants(min_times=2, start_date=None, end_date=None):
    """Candidates who sent their CV more than once (repeat applicants): how
    many people, how many repeat emails, and who sent the most. Use for 'how
    many candidates applied more than once', 'who sent their CV multiple
    times', 'duplicate / repeat CVs'."""
    min_times = _years(min_times, "min_times") or 2
    where, params = _date_where(start_date, end_date)
    where.append("status IN ('done', 'partial') AND dedup_key IS NOT NULL")
    w = " AND ".join(where)
    with closing(db.connect()) as conn:
        email = db.candidate_email_column(conn)
        per_person = conn.execute(
            f"SELECT m.dedup_key, COUNT(*) AS n, MAX(m.received_at) AS last, c.full_name, c.{email} "
            f"FROM (SELECT * FROM messages WHERE {w}) m LEFT JOIN candidates c ON c.dedup_key = m.dedup_key "
            f"GROUP BY m.dedup_key HAVING COUNT(*) >= ? ORDER BY n DESC, c.full_name",
            params + [min_times]).fetchall()
        repeat_emails = conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE {w} AND status = 'done' AND reason IN (?, ?)",
            params + list(REPEAT_REASONS)).fetchone()[0]
    people = len(per_person)
    rows = [{"Name": r[3] or "-", "Email": r[4] or r[0], "CVs sent": r[1], "Last sent": (r[2] or "")[:10]}
            for r in per_person[:MAX_ROWS]]
    times = "more than once" if min_times == 2 else f"{min_times} or more times"
    answer = f"**{people}** candidate(s) sent their CV {times}{_date_phrase(start_date, end_date)}."
    if people:
        if min_times == 2:
            answer += f" That is **{repeat_emails}** repeat emails in total."
        spread = Counter(r[1] for r in per_person)
        answer += "\n\n" + _bullets(f"Sent {n} times: {spread[n]} people" for n in sorted(spread))
        top = per_person[0]
        answer += f"\n\nMost CVs from one person: **{top[3] or top[4] or top[0]}** ({top[1]})."
    return answer, {"kind": "table", "title": f"{people} repeat applicant(s)", "answer": answer,
                    "rows": rows, "total": people,
                    "call": _call("repeat_applicants", min_times=min_times, start_date=start_date,
                                  end_date=end_date)}


TOOLS = [search_candidates, group_candidates, get_candidate,
         email_stats, failure_reasons, processing_timeline, repeat_applicants]
