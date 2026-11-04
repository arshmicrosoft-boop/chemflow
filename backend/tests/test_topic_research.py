"""Tests for turning a Claude response into a report. No API calls are made."""
from types import SimpleNamespace as NS

import pytest

from backend.app import topic_research
from backend.app.topic_research import ResearchError, _search_failures, assemble_report


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


PAPERS = [{"title": "Paper A", "year": 2020, "url": "https://a.org", "abstract": "Alpha."},
          {"title": "Paper B", "year": None, "url": "https://b.org", "abstract": "Beta."}]
LISTING = {"data": [
    {"id": "big/model:free", "context_length": 900, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "small/model:free", "context_length": 100, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "openrouter/free", "context_length": 50, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "paid/model", "context_length": 9999, "pricing": {"prompt": "0.001", "completion": "0.002"}},
    {"id": "paid/fake:free", "context_length": 9999, "pricing": {"prompt": "0.001", "completion": "0"}},
    {"id": "nameless/zero", "context_length": 9999, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "google/lyria-x:free", "context_length": 9999, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "broken/model:free", "context_length": 9999, "pricing": {}},
]}


def fake_get(monkeypatch, body=LISTING):
    monkeypatch.setattr(topic_research.requests, "get", lambda *a, **k: NS(json=lambda: body))


def fake_post(monkeypatch, status, body):
    response = NS(status_code=status, json=lambda: body)
    monkeypatch.setattr(topic_research.requests, "post", lambda *a, **k: response)


def choice(content="## Summary\nDone [1].", finish_reason="stop", **message):
    return {"choices": [{"finish_reason": finish_reason, "message": dict(content=content, **message)}]}


def test_only_free_text_models_are_offered(monkeypatch):
    fake_get(monkeypatch)
    assert topic_research.free_models() == ["openrouter/free", "big/model:free", "small/model:free"]
    assert topic_research.free_models("small/model:free")[0] == "small/model:free"


def test_a_paid_preferred_model_is_refused(monkeypatch):
    fake_get(monkeypatch)
    with pytest.raises(ResearchError) as e:
        topic_research.free_models("paid/model")
    assert "free model" in str(e.value)


def test_request_never_uses_the_paid_web_plugin(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(json)
        return NS(status_code=200, json=lambda: choice())

    monkeypatch.setattr(topic_research.requests, "post", post)
    topic_research._ask_model("big/model:free", "prompt")
    assert "plugins" not in sent and "tools" not in sent


def test_report_keeps_paper_numbers_as_sources():
    out = topic_research.assemble_paper_report("Sure!\n## Summary\nAlpha [1].", PAPERS)
    assert out["report"] == "## Summary\nAlpha [1]."
    assert out["sources"] == [{"url": "https://a.org", "title": "Paper A"},
                              {"url": "https://b.org", "title": "Paper B"}]


def test_uncited_report_is_rejected():
    with pytest.raises(ResearchError):
        topic_research.assemble_paper_report("## Summary\nNo citations.", PAPERS)
    with pytest.raises(ResearchError):
        topic_research.assemble_paper_report("## Summary\nInvented [9].", PAPERS)


def test_abstract_is_rebuilt_from_openalex_positions():
    assert topic_research._reconstruct_abstract({"world": [1], "hello": [0, 2]}) == "hello world hello"


def test_papers_are_deduplicated_and_empty_abstracts_dropped(monkeypatch):
    monkeypatch.setattr(topic_research, "_openalex_papers", lambda t: PAPERS)
    monkeypatch.setattr(topic_research, "_europepmc_papers", lambda t: [
        dict(PAPERS[0], title="PAPER  a!"),
        {"title": "No abstract", "year": 1, "url": "u", "abstract": " "},
    ])
    assert [p["title"] for p in topic_research.find_papers("t")] == ["Paper A", "Paper B"]


def test_one_database_failing_is_tolerated(monkeypatch):
    def down(topic):
        raise topic_research.requests.ConnectionError()

    monkeypatch.setattr(topic_research, "_openalex_papers", down)
    monkeypatch.setattr(topic_research, "_europepmc_papers", lambda t: PAPERS)
    assert len(topic_research.find_papers("t")) == 2
    monkeypatch.setattr(topic_research, "_europepmc_papers", down)
    with pytest.raises(ResearchError):
        topic_research.find_papers("t")


def setup_switching(monkeypatch, outcomes):
    tried = []

    def ask(model, prompt):
        tried.append(model)
        outcome = outcomes[len(tried) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(topic_research, "find_papers", lambda t: PAPERS)
    monkeypatch.setattr(topic_research, "free_models", lambda p="": ["one:free", "two:free"])
    monkeypatch.setattr(topic_research, "_ask_model", ask)
    return tried


def test_switches_to_the_next_free_model(monkeypatch):
    tried = setup_switching(monkeypatch, [ResearchError("busy", status=429), "## Summary\nDone [2]."])
    assert topic_research._research_openrouter("t")["model"] == "two:free"
    assert tried == ["one:free", "two:free"]


def test_free_router_is_retried_before_other_models(monkeypatch):
    tried = setup_switching(monkeypatch, [ResearchError("slow", status=504)] * 3 + ["## Summary\nDone [1]."])
    monkeypatch.setattr(topic_research, "free_models", lambda p="": ["openrouter/free", "one:free"])
    assert topic_research._research_openrouter("t")["model"] == "one:free"
    assert tried == ["openrouter/free"] * 3 + ["one:free"]


def test_preferred_free_model_goes_first(monkeypatch):
    tried = setup_switching(monkeypatch, ["## Summary\nDone [1]."])
    monkeypatch.setattr(topic_research, "free_models", lambda p="": [p, "openrouter/free"])
    assert topic_research._research_openrouter("t", "nvidia/x:free")["model"] == "nvidia/x:free"
    assert tried == ["nvidia/x:free"]


def test_uncited_answer_also_switches_model(monkeypatch):
    setup_switching(monkeypatch, ["## Summary\nNo citations.", "## Summary\nDone [1]."])
    assert topic_research._research_openrouter("t")["model"] == "two:free"


@pytest.mark.parametrize("status", [401, 403, 422])
def test_key_and_safety_failures_do_not_switch(monkeypatch, status):
    tried = setup_switching(monkeypatch, [ResearchError("stop", status=status), "## Summary\nDone [1]."])
    with pytest.raises(ResearchError):
        topic_research._research_openrouter("t")
    assert tried == ["one:free"]


def test_all_free_models_failing_reports_the_last_error(monkeypatch):
    setup_switching(monkeypatch, [ResearchError("a", status=429), ResearchError("b", status=502)])
    with pytest.raises(ResearchError) as e:
        topic_research._research_openrouter("t")
    assert str(e.value) == "b"


@pytest.mark.parametrize("status, body, expected_status, words", [
    (401, {"error": {"code": 401, "message": "x"}}, 401, "rejected"),
    (429, {"error": {"code": 429, "message": "x"}}, 429, "free"),
    (400, {"error": {"code": 400, "message": "not a valid model"}}, 400, "not a valid model"),
    (200, {"error": {"code": 502, "message": "x"}}, 502, "down"),
    (200, {"choices": []}, 502, "empty"),
    (200, choice(finish_reason="content_filter"), 422, "declined"),
    (200, choice(refusal="no"), 422, "declined"),
    (200, choice(finish_reason="length"), 502, "cut off"),
    (200, choice(content=""), 502, "empty report"),
])
def test_openrouter_failures_are_explained(monkeypatch, status, body, expected_status, words):
    fake_post(monkeypatch, status, body)
    with pytest.raises(ResearchError) as e:
        topic_research._ask_model("some/model:free", "prompt")
    assert e.value.status == expected_status and words in str(e.value)


@pytest.mark.parametrize("anthropic_key, openrouter_key, provider, model", [
    ("", "", None, None),
    ("a", "", "anthropic", topic_research.MODEL),
    ("", "o", "openrouter", "free models"),
    ("a", "o", "anthropic", topic_research.MODEL),
])
def test_provider_follows_the_keys(monkeypatch, anthropic_key, openrouter_key, provider, model):
    monkeypatch.setattr(topic_research, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    monkeypatch.setenv("OPENROUTER_API_KEY", openrouter_key)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert topic_research.active_provider() == provider
    assert topic_research.active_model() == model


def setup_chat(monkeypatch, outcomes, key="o"):
    tried = []

    def complete(model, messages, timeout):
        tried.append((model, messages))
        outcome = outcomes[len(tried) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(topic_research, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setattr(topic_research, "free_models", lambda p="": ["one:free", "two:free"])
    monkeypatch.setattr(topic_research, "_complete", complete)
    return tried


def test_chat_keeps_the_conversation_and_answers(monkeypatch):
    tried = setup_chat(monkeypatch, [" Hello. "])
    history = [{"role": "user", "content": "Hi"}]
    assert topic_research.chat(history) == {"reply": "Hello.", "model": "one:free"}
    assert [m["role"] for m in tried[0][1]] == ["system", "user"]


def test_chat_switches_to_the_next_free_model(monkeypatch):
    setup_chat(monkeypatch, [ResearchError("busy", status=429), "Hello."])
    assert topic_research.chat([{"role": "user", "content": "Hi"}])["model"] == "two:free"


def test_chat_does_not_switch_on_a_bad_key(monkeypatch):
    tried = setup_chat(monkeypatch, [ResearchError("bad key", status=401), "Hello."])
    with pytest.raises(ResearchError):
        topic_research.chat([{"role": "user", "content": "Hi"}])
    assert len(tried) == 1


def test_chat_explains_a_missing_key(monkeypatch):
    setup_chat(monkeypatch, [], key="")
    with pytest.raises(ResearchError) as e:
        topic_research.chat([{"role": "user", "content": "Hi"}])
    assert e.value.status == 503
