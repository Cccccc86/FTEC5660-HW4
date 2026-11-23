"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from rank_bm25 import BM25Okapi

load_dotenv()


_known_companies: set[str] = set()

_YEAR_IN_QUERY = re.compile(r"(?i)\b(?:FY\s*[-–]?\s*)?((?:19|20)\d{2})\b")
_CALL_QUERY = re.compile(r"\b(?:earnings|conference|results|fourth[- ]quarter|q[1-4])[- ]+call\b", re.I)
_RESEARCH_QUERY = re.compile(
    r"\b(?:broker(?:age)?|equity research|research report|results review|"
    r"target price|price target|investment rating|recommendation|"
    r"forecast(?:s|ed)?|consensus estimate)\b|"
    r"\brating\b[^?;]*\b(?:give[sn]?|gave|assign(?:s|ed)?|set[st]?|from)\b",
    re.I,
)
_COMPARISON_QUERY = re.compile(
    r"\b(?:which\s+(?:company|firm|business|of)|compare|versus|vs)\b|"
    r"\b(?:which|did)\b[^?;]*\b(?:higher|lower|more|less|greater|fewer)\b",
    re.I,
)

# Case identifies names absent from the library, while names actually seen in
# the corpus can also be matched when a question is written in lower case.
_NAME_WORD = (
    r"(?!(?:Who|What|Which|Where|When|How|Did|Does|Do|Is|Are|Was|Were|Has|Have|"
    r"According|In|On|At|For|Between|"
    r"Compared|Compare|From|USD|HKD|EUR|FY\d{4}|FY|Q[1-4])\b)"
    r"[A-Z][^\W_]*(?:[-'][A-Z0-9][^\W_]*)*"
)
_PROPER_NAME = re.compile(
    rf"(?<![\w'-]){_NAME_WORD}(?:(?:\s+|\s*&\s*|\s+of\s+|\s+of the\s+){_NAME_WORD})*"
)
_QUERY_SUFFIX = re.compile(r"(?:,?\s+(?:Inc|Corp|Corporation|Co|Ltd|Limited|PLC|LLC)\.?)+$", re.I)
_POSSESSIVE = re.compile(r"^\.?'(?:s\b|(?!\w))", re.I)
_COMPANY_SUBJECT = re.compile(
    r"^\.?\s+(?:report(?:s|ed)?|hold(?:s)?|held|earn(?:s|ed)?|generat(?:e|es|ed)|"
    r"disclos(?:e|es|ed)|announc(?:e|es|ed)|record(?:s|ed)?|post(?:s|ed)?|"
    r"incur(?:s|red)?|pay(?:s)?|paid|spend|spent|hav(?:e|ing)|has|had|is|was|were|"
    r"audited|located|headquartered)\b",
    re.I,
)
_BROKER_ACTION = re.compile(
    r"^\s+(?:forecast|estimate|project|predict|assign|set|give|rate|recommend|"
    r"publish|cover|upgrade|downgrade)(?:s|d|ed)?\b", re.I
)


def _query_name_key(name: str) -> str:
    return " ".join(_QUERY_SUFFIX.sub("", name).casefold().split())


def _query_mentions(question: str) -> list[tuple[int, int, str, bool]]:
    """Find candidate names and mark those supported by the corpus metadata."""
    known_keys = {_query_name_key(name) for name in _known_companies}
    mentions = []
    for match in _PROPER_NAME.finditer(question):
        name = match.group()
        mentions.append((match.start(), match.end(), name, _query_name_key(name) in known_keys))

    for key in sorted(known_keys):
        if not key:
            continue
        words = r"\s+".join(re.escape(word) for word in key.split())
        for match in re.finditer(rf"(?<!\w){words}(?!\w)", question, re.I):
            if any(start <= match.start() and end >= match.end() for start, end, *_ in mentions):
                continue
            mentions.append((match.start(), match.end(), match.group(), True))
    return sorted(mentions)


def _is_company_mention(
    question: str, start: int, end: int, known: bool, doc_type: str, comparison: bool
) -> bool:
    left, right = question[:start], question[end:]
    if re.search(r"\b(?:audited|reviewed)\s+by\s*$", left, re.I):
        return False
    if doc_type == "research_report":
        attributed = re.search(r"\b(?:according to|(?:reported|published|issued) by)\s*$", left, re.I)
        if _BROKER_ACTION.match(right) or attributed:
            return False
        if _POSSESSIVE.match(right) and re.search(
            r"^\.?'(?:s\b)?\s*(?:(?:FY\s*)?\d{4}\s+)?(?:results review|research report)\b|"
            r"^\.?'(?:s\b)?\s*[^,?;]{0,60}\b(?:forecast|estimate|rating|recommendation|target price)\s+for\b",
            right, re.I,
        ):
            return False

    if known or _POSSESSIVE.match(right) or _COMPANY_SUBJECT.match(right):
        return True
    if comparison and (
        re.search(r"(?:[:,(]|\b(?:or|and|of|than|versus|vs|did|compare)\b\.?)\s*$", left, re.I)
        and re.match(r"\s*(?:[?.,;)]|$|\b(?:or|and|versus|vs)\b)", right, re.I)
    ):
        return True
    return bool(re.search(
        r"\b(?:for|of|at|did|does|do|audited|give|gives|gave|rate|rates|rated|cover|covers)\s*$",
        left, re.I,
    ) and re.match(r"^\s*(?:[?.,;]|$|\b(?:in|as|at|on|for|during|from|before|after)\b)", right, re.I))


def parse_query(question: str) -> dict:
    """Identify the companies as worded, named fiscal year, and requested document type."""
    question = unicodedata.normalize("NFKC", question).replace("’", "'")
    year = _YEAR_IN_QUERY.search(question)
    if _CALL_QUERY.search(question):
        doc_type = "earnings_call"
    elif _RESEARCH_QUERY.search(question):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    companies: dict[str, str] = {}
    comparison = bool(_COMPARISON_QUERY.search(question))
    for start, end, name, known in _query_mentions(question):
        if _is_company_mention(question, start, end, known, doc_type, comparison):
            key = _query_name_key(name)
            if key:
                companies.setdefault(key, name)
    return {
        "companies": list(companies.values()),
        "fiscal_year": int(year.group(1)) if year else None,
        "doc_type": doc_type,
    }


def _visible_text(tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split()) if tag else ""


def _row_cells(row):
    return row.find_all(["th", "td"], recursive=False)


def table_to_text(table_html: str) -> str:
    """Read each table row with its own label, column labels and original values."""
    table = BeautifulSoup(table_html, "html.parser").find("table")
    if table is None:
        return ""
    caption = _visible_text(table.find("caption", recursive=False))
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    if not rows:
        return caption

    header = table.find("thead", recursive=False)
    header_rows = header.find_all("tr") if header else []
    if not header_rows and all(cell.name == "th" for cell in _row_cells(rows[0])):
        header_rows = rows[:1]
    columns = []
    if header_rows:
        for cell in _row_cells(header_rows[-1]):
            columns.extend([_visible_text(cell)] * int(cell.get("colspan", 1)))

    lines = [caption] if caption else []
    for row in rows:
        if any(row is header_row for header_row in header_rows):
            continue
        cells = _row_cells(row)
        if not cells:
            continue
        label = _visible_text(cells[0])
        if len(cells) == 1:
            if label:
                lines.append(label)
            continue

        fields = []
        column = int(cells[0].get("colspan", 1))
        for cell in cells[1:]:
            value = _visible_text(cell)
            span = int(cell.get("colspan", 1))
            headings = list(dict.fromkeys(h for h in columns[column:column + span] if h))
            column += span
            if value:
                heading = " / ".join(headings)
                fields.append(f"{heading}: {value}" if heading else value)

        # In a wide table, continue on a new line with the same row label.
        line = label
        for field in fields:
            candidate = f"{line} | {field}" if line else field
            if len(candidate) > MAX_LINE_CHARS and line != label:
                lines.append(line)
                line = f"{label} | {field}" if label else field
            else:
                line = candidate
        if line:
            lines.append(line)
    return "\n".join(lines)


def document_metadata(file_name: str, soup: BeautifulSoup) -> dict:
    """Read the document date and the company under coverage from its cover."""
    company = _visible_text(soup.find("h1"))
    subtitle = _visible_text(soup.select_one(".subtitle"))
    first_heading = _visible_text(soup.find("h2"))
    title = _visible_text(soup.title)
    broker_line = _visible_text(soup.select_one(".broker"))
    header = "\n".join((subtitle, first_heading, title))
    if re.search(r"\bresults\s+review\b", header, re.I) or broker_line:
        doc_type = "research_report"
        date_pattern = r"\b(?:FY\s*)?((?:19|20)\d{2})\s+results\s+review\b"
    elif re.search(r"\bearnings(?:\s+conference)?\s+call\b", header, re.I):
        doc_type = "earnings_call"
        date_pattern = r"\b(?:full\s+year|FY)\s*((?:19|20)\d{2})\b"
    else:
        doc_type = "annual_report"
        date_pattern = r"\bannual\s+report\s+(?:FY\s*)?((?:19|20)\d{2})\b"
    if doc_type != "annual_report":
        # These two source formats append an exchange/ticker annotation to h1.
        company = re.sub(r"\s+\([^()]+\)\s*$", "", company)
    year = re.search(date_pattern, header, re.I)
    if year is None:
        year = re.search(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", header, re.I)

    if company:
        _known_companies.add(company)
    metadata = {
        "source": file_name,
        "company": company,
        "fiscal_year": int(year[1]) if year else None,
        "doc_type": doc_type,
    }
    if broker_line:
        metadata["broker"] = broker_line.split("|", 1)[0].strip()
    return metadata


def _pack_passages(parts: list[str], heading: str = "") -> list[str]:
    """Pack complete paragraphs or table rows, repeating the source heading."""
    prefix = heading.strip()[:200]
    room = MAX_CHUNK_CHARS - len(prefix) - bool(prefix)
    passages = []
    current = []
    size = 0
    for text in parts:
        while len(text) > room:
            boundary = text.rfind(" ", 0, room + 1)
            split = boundary if boundary > room // 2 else room
            if current:
                passages.append("\n".join(current))
                current, size = [], 0
            passages.append(text[:split])
            text = text[split:].lstrip()
        if text:
            if current and size + 1 + len(text) > room:
                passages.append("\n".join(current))
                current, size = [], 0
            size += len(text) + bool(current)
            current.append(text)
    if current:
        passages.append("\n".join(current))
    return [f"{prefix}\n{passage}" if prefix else passage for passage in passages]


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Keep tables separate and keep related paragraphs under their own heading."""
    soup = BeautifulSoup(html, "html.parser")
    metadata = document_metadata(file_name, soup)
    body = soup.body or soup
    block_names = ("h1", "h2", "h3", "h4", "p", "li", "table", "blockquote", "pre")
    section = ""
    paragraphs: list[str] = []
    chunks: list[Document] = []

    def emit(parts: list[str], *, caption: str = "", is_table: bool = False) -> None:
        header = "\n".join(dict.fromkeys(filter(None, (metadata.get("broker"), section, caption))))
        chunk_metadata = {**metadata, "section": section, "is_table": is_table}
        if caption:
            chunk_metadata["caption"] = caption
        for passage in _pack_passages(parts, header):
            chunks.append(Document(page_content=passage, metadata=chunk_metadata))

    for block in body.find_all(block_names):
        if block.find_parent((*block_names, "nav", "footer")):
            continue
        if block.name in ("h1", "h2", "h3", "h4"):
            emit(paragraphs)
            paragraphs = []
            section = _visible_text(block)
        elif block.name == "table":
            emit(paragraphs)
            paragraphs = []
            caption = _visible_text(block.find("caption", recursive=False))
            rows = table_to_text(str(block)).splitlines()
            emit(rows[1:] if caption else rows, caption=caption, is_table=True)
        else:
            value = _visible_text(block)
            if value:
                paragraphs.append(value)
    emit(paragraphs)
    return chunks


@dataclass
class SearchIndex:
    chunks: list[Document]
    bm25: BM25Okapi | None
    by_company: dict[str, list[int]]
    variant: int


# Terms that describe the question rather than the fact being looked up.
_QUESTION_WORDS = set('''
    what which who where how when is are was were did does do in on at as from
    for of the a an its their and or s had has have company companies held hold
    more higher lower greater less most fewest much many value amount number
    end year fiscal ended financial usd millions million billion percent
    percentage dollars dec december usd fy report reported there own
'''.split())


def _search_tokens(text: str) -> list[str]:
    """Split punctuation and FY-prefixed dates without losing the year."""
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'\bfy\s*(20\d{2})[ae]?\b', r'\1', text)
    return re.findall(r"[a-z]+|\d+(?:[,.]\d+)*", text)


def _make_search_index(chunks: list[Document], variant: int) -> SearchIndex:
    tokens = [c.page_content.lower().split() for c in chunks]
    groups: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        key = company_key(chunk.metadata['company'])
        groups.setdefault(key, []).append(i)
    return SearchIndex(chunks, BM25Okapi(tokens) if any(tokens) else None, groups, variant)


def build_index(chunks: list[Document]) -> SearchIndex:
    """Index the supplied corpus in memory for lexical retrieval."""
    return _make_search_index(chunks, variant=6)


def _report_year(question: str, kind: str) -> int | None:
    """A year attached to a source title can differ from the forecast year."""
    source = {
        'earnings_call': r'(?:earnings(?: conference)?|results) call',
        'research_report': r'(?:results review|research report)',
    }.get(kind)
    if not source:
        return None
    year = re.search(rf'\b((?:19|20)\d{{2}}) (?:q [1-4] )?{source}\b', _plain(question))
    return int(year[1]) if year else None


def _scope_for_company(index: SearchIndex, key: str, parsed: dict, report_year: int | None) -> list[int]:
    kind = parsed['doc_type']
    year = parsed['fiscal_year']
    scoped = []
    for i in index.by_company.get(key, []):
        meta = index.chunks[i].metadata
        if meta['doc_type'] != kind:
            continue
        if report_year is not None and meta['fiscal_year'] != report_year:
            continue
        if kind == 'annual_report' and year is not None:
            if meta['fiscal_year'] not in (year, year + 1):
                continue
        scoped.append(i)
    return scoped


def _phrase(text: str, phrase: str) -> bool:
    return f' {phrase} ' in f' {text} '


def _plain(text: str) -> str:
    return ' '.join(_search_tokens(text))


def _row_texts(chunk: Document) -> list[tuple[str, str]]:
    return [(line.split(' | ', 1)[0], line) for line in chunk.page_content.splitlines()
            if ' | ' in line]


_STATEMENT_FIELDS = (
    (r'net accounts receivable|accounts receivable net|net trade receivables|accounts receivable',
     r'accounts receivable(?: net)?|(?:net )?trade receivables(?: net)?|net accounts receivable'),
    (r'cash and cash equivalents|cash equivalents|cash holdings', r'cash and cash equivalents'),
    (r'net cash (?:from|provided by) operating activities|operating cash flow',
     r'net cash (?:from|provided by) operating activities|operating cash flow'),
    (r'long term (?:borrowing[s]?|debt)', r'long term (?:borrowings?|debt)'),
    (r'short term (?:borrowing[s]?|debt)', r'short term (?:borrowings?|debt)'),
    (r'total assets', r'total assets'),
    (r'total liabilities', r'total liabilities'),
    (r'total (?:shareholders? )?equity', r'total (?:shareholders? )?equity'),
    (r'inventor(?:y|ies)', r'inventor(?:y|ies)'),
    (r'net income|net profit', r'net (?:income|profit)'),
    (r'gross profit', r'gross profit'),
    (r'operating (?:income|profit)', r'operating (?:income|profit)'),
    (r'employees|headcount|number of people', r'total employees|headcount|total headcount'),
    (r'revenues?|sales', r'(?:total )?(?:revenues?|sales)'),
)


def _question_focus(question: str, parsed: dict) -> str:
    focus = _plain(question)
    for name in sorted(parsed['companies'], key=len, reverse=True):
        name_text = _plain(name)
        focus = re.sub(rf'\b{re.escape(name_text)}\b', ' ', focus)
    return ' '.join(focus.split())


def _field_rules(focus: str) -> tuple[str | None, str | None]:
    """Pick a row family only when the question asks for that field itself."""
    if re.search(r'\b(?:revenues?|sales)\b', focus):
        if re.search(r'\bsegment\b', focus):
            return 'segment', None
        if re.search(r'\bfrom\b|\bgenerat(?:e|ed) in [a-z]', focus):
            return 'split', None
    for phrase, field in _STATEMENT_FIELDS:
        if re.search(rf'\b(?:{phrase})\b', focus):
            return 'statement', field
    return None, None


def _has_year_value(line: str, year: int | None) -> bool:
    if year is None:
        return True
    return any(re.search(rf'\b{year}\b', _plain(cell.split(':', 1)[0]))
               for cell in line.split(' | ')[1:] if ':' in cell)


def _evidence_score(chunk: Document, focus: str, parsed: dict) -> int:
    """Boost a matching row or passage that names the requested fact."""
    content = chunk.page_content
    plain = _plain(content)
    kind = parsed['doc_type']
    year = parsed['fiscal_year']
    meta = chunk.metadata
    rows = _row_texts(chunk)
    if kind == 'annual_report':
        if re.search(r'\bchief (?:financial|executive|operating) officer\b|\b(?:cfo|ceo|coo)\b', focus):
            if year is not None and meta['fiscal_year'] != year:
                return 0
            title = re.search(r'\bchief (?:financial|executive|operating) officer\b', focus)
            role = title[0] if title else next(
                role for acronym, role in [('cfo', 'chief financial officer'),
                                           ('ceo', 'chief executive officer'),
                                           ('coo', 'chief operating officer')] if _phrase(focus, acronym))
            if any(_phrase(_plain(line), role) for _, line in rows):
                return 5 if 'executive officers' in plain else 3
            return 2 if _phrase(plain, role) else 0
        if re.search(r'\baudit(?:or|ed| firm)?\b', focus):
            if year is not None and meta['fiscal_year'] != year:
                return 0
            return 5 if re.search(r'\bindependent audit(?:or| firm)\b|\baudited by\b', plain) else 0
        if re.search(r'\bheadquarter(?:s|ed)?\b|\bhead office\b', focus):
            return 5 if re.search(r'\bheadquarter(?:s|ed)\b|\bhead office\b', plain) else 0

        family, label = _field_rules(focus)
        if family == 'statement' and label:
            if any(re.fullmatch(label, _plain(row)) and _has_year_value(line, year) for row, line in rows):
                return 6 if year is not None and meta['fiscal_year'] == year else 5
            return 0
        if family in ('segment', 'split'):
            if 'revenue by segment' not in plain and (family == 'segment' or 'revenue by region' not in plain):
                return 0
            if any(_phrase(focus, _plain(row)) and _has_year_value(line, year)
                   for row, line in rows if _plain(row) not in {'total', 'total revenue', 'revenue'}):
                return 6 if year is not None and meta['fiscal_year'] == year else 5
            return 0

    if kind == 'research_report':
        if re.search(r'\b(?:target price|price target)\b', focus):
            return 6 if re.search(r'\b(?:target price|price target)\b', plain) and re.search(r'\d', content) else 0
        if re.search(r'\brating\b|\brecommendation\b', focus):
            return 6 if re.search(r'\brating\s+(?:buy|hold|reduce|sell|neutral|outperform|underperform)\b', plain) else 0
        if re.search(r'\bforecast\b|\bestimat', focus):
            _, label = _field_rules(focus)
            if label and any(re.fullmatch(label, _plain(row)) and _has_year_value(line, year)
                             for row, line in rows):
                return 6

    if kind == 'earnings_call':
        if re.search(r'\brepurchase\b|\bbuyback\b', focus):
            return 6 if re.search(r'\brepurchase\b|\bbuyback\b', plain) and re.search(r'\d', content) else 0
        if re.search(r'\bguid\w+\b|\boutlook\b', focus):
            year_ok = year is None or str(year) in _search_tokens(content)
            return 6 if year_ok and re.search(r'\bguid\w+\b|\bexpect\w+\b|\boutlook\b', plain) else 0

    # Unknown fields need actual words from the question, in the same sentence or row.
    terms = set(_search_tokens(focus)) - _QUESTION_WORDS
    terms = {word for word in terms if word.isalpha() and len(word) >= 3}
    if not terms:
        return 0
    return 1 if any(terms <= set(_search_tokens(line)) for line in content.splitlines()) else 0


def _select_chunks(index: SearchIndex, question: str) -> list[Document]:
    if index.bm25 is None:
        return []
    variant = index.variant
    query_tokens = question.lower().split()
    lexical = index.bm25.get_scores(query_tokens)
    all_indices = list(range(len(index.chunks)))
    if variant < 3:
        selected = sorted(all_indices, key=lambda i: -lexical[i])[:MAX_RETRIEVED]
        return [index.chunks[i] for i in selected]

    parsed = parse_query(question)
    keys = list(dict.fromkeys(company_key(name) for name in parsed['companies'] if name.strip()))
    source_year = _report_year(question, parsed['doc_type'])
    groups = [_scope_for_company(index, key, parsed, source_year) for key in keys]
    if variant >= 6 and (not groups or any(not group for group in groups)):
        return []
    scoped = list(dict.fromkeys(i for group in groups for i in group)) or all_indices
    focus = _question_focus(question, parsed)
    scores = {i: _evidence_score(index.chunks[i], focus, parsed) if variant >= 4 else 0
              for i in scoped}
    if variant >= 6:
        groups = [[i for i in group if scores[i] > 0] for group in groups]
        if any(not group for group in groups):
            return []
        scoped = [i for group in groups for i in group]

    def order(pool: list[int]) -> list[int]:
        return sorted(pool, key=lambda i: (-scores[i], -lexical[i]))

    ranked = order(scoped)
    if variant >= 5 and len(groups) > 1 and all(groups):
        # Reserve the first available slot for each company in a comparison.
        first = [order(group)[0] for group in groups]
        best = first + [i for i in ranked if i not in first]
    else:
        best = ranked
    return [index.chunks[i] for i in best[:MAX_RETRIEVED]]


def retrieve(index: SearchIndex, question: str) -> list[Document]:
    """Return up to five original chunks that support the requested fact(s)."""
    return _select_chunks(index, question)


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
