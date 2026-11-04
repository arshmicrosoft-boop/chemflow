"""Tests for turning a Claude response into a report. No API calls are made."""
from types import SimpleNamespace as NS

from backend.app.topic_research import _search_failures, assemble_report


def cite(url, title="Paper"):
    return NS(url=url, title=title, cited_text="quote")


def text(body, citations=None):
    return NS(type="text", text=body, citations=citations)


def test_citations_become_numbered_sources():
    out = assemble_report([
        text("## Summary\n"),
        text("Lithium is light.", [cite("https://a.org"), cite("https://b.org", None)]),
        text(" It also reacts.\n", [cite("https://a.org")]),
    ])
    assert out["report"] == "## Summary\nLithium is light.[1][2] It also reacts.[1]"
    assert out["sources"] == [
        {"url": "https://a.org", "title": "Paper"},
        {"url": "https://b.org", "title": "https://b.org"},
    ]


def test_preamble_and_non_text_blocks_are_dropped():
    out = assemble_report([
        NS(type="thinking", thinking=""),
        NS(type="server_tool_use", name="web_search"),
        NS(type="server_tool_use", name="web_search"),
        text("I will look into it.\n\n"),
        text("## Summary\nDone."),
    ])
    assert out["report"] == "## Summary\nDone."
    assert out["searches"] == 2


def test_report_without_headings_is_kept():
    assert assemble_report([text("Plain answer.")])["report"] == "Plain answer."


def test_search_errors_are_found():
    assert _search_failures([
        NS(type="web_search_tool_result", content=[NS(url="https://a.org")]),
        NS(type="web_search_tool_result", content=NS(error_code="max_uses_exceeded")),
        text("x"),
    ]) == ["max_uses_exceeded"]
