# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.
"""Unit tests for two-tier URI routing (Tier 1 deterministic + Tier 2 bounded `roles.decision`)."""
from __future__ import annotations

from typing import Any

from urirun_connector_router import build_decision_payload, decide_uri_route
from urirun_connector_router.decision_fallback import (
    DECISION_SCHEMA,
    NO_MATCH_LABEL,
    is_guarded_query,
    normalize_candidates,
)


_CATALOG = [
    {
        "uri": "router://host/route/query/resolve",
        "description": "Rozwiąż gdzie uruchomi się wskazana trasa URI",
        "triggers": ["resolve", "trasa", "lokalizacja"],
    },
    {
        "uri": "router://host/plan/query/diagnose",
        "description": "Zdiagnozuj wszystkie kroki planu URI i zablokowane warstwy",
        "triggers": ["diagnose", "plan", "kroki"],
    },
    {
        "uri": "router://host/mesh/query/targets",
        "description": "Wylistuj dostępne węzły i cele wykonania w sieci mesh",
        "triggers": ["targets", "mesh", "wezly"],
    },
]


class _StubDecisionClient:
    def __init__(self, answer: str | None = None, *, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.profile = {"roles": {"decision": {"model": "mesh/d1"}}}

    def infer(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((role, payload))
        if self.error is not None:
            raise self.error
        return {"model": "mesh/d1", "answers": {"route": self.answer}}


def test_tier1_exact_uri_and_unambiguous_match_skip_llm() -> None:
    client = _StubDecisionClient("op_0")
    exact = decide_uri_route("router://host/mesh/query/targets", _CATALOG, client=client)
    assert exact["schema"] == DECISION_SCHEMA
    assert exact["status"] == "selected"
    assert exact["tier"] == "tier1_exact_uri"
    assert exact["selected_uri"] == "router://host/mesh/query/targets"
    assert client.calls == []

    det = decide_uri_route("wylistuj dostepne wezly mesh", _CATALOG, client=client)
    assert det["status"] == "selected"
    assert det["tier"] == "tier1_deterministic"
    assert det["selected_uri"] == "router://host/mesh/query/targets"
    assert client.calls == []


def test_guarded_queries_fail_closed_without_calling_decision_model() -> None:
    client = _StubDecisionClient("op_0")
    for text in (
        "",
        "Czy mogę uruchomić plan?",
        "Nie uruchamiaj diagnozy planu",
        "Sformatuj dysk i zniszcz system",
    ):
        assert is_guarded_query(text) is True
        res = decide_uri_route(text, _CATALOG, client=client)
        assert res["status"] == "no_match"
        assert res["tier"] == "guard"
        assert res["selected_uri"] is None
    assert client.calls == []


def test_tier2_bounded_decision_resolves_ambiguous_query_via_roles_decision() -> None:
    client = _StubDecisionClient("op_1")
    res = decide_uri_route("sprawdz problemy w procedurze uruchomieniowej", _CATALOG, client=client)
    assert res["status"] == "selected"
    assert res["tier"] == "tier2_decision"
    assert res["selected_uri"] == "router://host/plan/query/diagnose"
    assert res["model"] == "mesh/d1"
    assert len(client.calls) == 1
    role, payload = client.calls[0]
    assert role == "decision"
    assert payload["offline"] is True
    assert payload["questions"][0]["options"]["NO_MATCH"] == NO_MATCH_LABEL


def test_tier2_rejects_hallucinated_or_unknown_option_and_handles_no_match() -> None:
    for answer in ("NO_MATCH", "op_99", "router://evil/invented/route", None):
        client = _StubDecisionClient(answer)
        res = decide_uri_route("sprawdz dziwna operacje poza katalogiem", _CATALOG, client=client)
        assert res["status"] == "no_match"
        assert res["selected_uri"] is None


def test_normalize_candidates_filters_invalid_uris_and_dedupes() -> None:
    norm = normalize_candidates([
        {"uri": "not-a-uri", "description": "bad"},
        {"uri": "router://host/mesh/query/targets", "description": "first"},
        {"uri": "router://host/mesh/query/targets", "description": "duplicate"},
    ])
    assert len(norm) == 1
    assert norm[0]["uri"] == "router://host/mesh/query/targets"
    payload, mapping = build_decision_payload("test", norm)
    assert mapping == {"op_0": "router://host/mesh/query/targets"}
    assert set(payload["questions"][0]["options"]) == {"op_0", "NO_MATCH"}


def test_tier2_gateway_error_and_unconfigured_fail_open(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HEYASA_AI_CONFIG", str(tmp_path / "missing.json"))
    res_missing = decide_uri_route("niejednoznaczne zapytanie operacyjne", _CATALOG)
    assert res_missing["status"] == "unavailable"
    assert res_missing["selected_uri"] is None

    client = _StubDecisionClient(error=RuntimeError("timeout"))
    res_err = decide_uri_route("niejednoznaczne zapytanie operacyjne", _CATALOG, client=client)
    assert res_err["status"] == "unavailable"
    assert res_err["reason"] == "RuntimeError"
    assert res_err["selected_uri"] is None
