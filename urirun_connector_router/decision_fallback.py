# Author: Tom Sapletta · https://tom.sapletta.com
# Part of the ifURI solution.
"""Two-tier URI routing: Tier 1 deterministic UriRouter + Tier 2 bounded `roles.decision` (`/v1/decide`).

Tier 1 resolves exact URIs and unambiguous lexical matches in sub-millisecond time
without touching any LLM. Ambiguous queries fall through to Tier 2, which sends a
bounded choice prompt with symmetric labels to `roles.decision` (`mesh/d1`) and
validates the selected URI strictly against the declared candidate allowlist.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Mapping, Sequence

from urirun_connector_router.routing import parse_uri

DECISION_SCHEMA = "urirun.connector-router.decision/v1"
DEFAULT_DECISION_MODEL = "mesh/d1"
MAX_CANDIDATES = 12
NO_MATCH_LABEL = "żadna z dostępnych tras URI"

_GUARD_NEGATION_RE = re.compile(r"^(?:nie|dont|do not|never)\b", re.IGNORECASE)
_GUARD_QUESTION_RE = re.compile(
    r"^(?:czy|dlaczego|czemu|jak|kiedy|gdzie|po co|co to|why|how|when|where|can|could|should|what|is|are)\b",
    re.IGNORECASE,
)
_GUARD_DESTRUCTIVE_RE = re.compile(
    r"\b(?:sformatuj|zniszcz|rm\s+-rf|drop\s+database|format\s+disk)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _fold(text: str) -> str:
    cleaned = text.lower().replace("ł", "l").replace("'", "")
    return "".join(c for c in unicodedata.normalize("NFKD", cleaned) if not unicodedata.combining(c)).strip()


def _tokens(text: str) -> set[str]:
    return {tok for tok in _TOKEN_RE.findall(_fold(text)) if len(tok) >= 2}


def _stem_overlap(query_tokens: set[str], target_tokens: set[str]) -> float:
    if not query_tokens or not target_tokens:
        return 0.0
    target_stems = {tok[:4] for tok in target_tokens if len(tok) >= 4}
    score = 0.0
    for tok in query_tokens:
        if tok in target_tokens:
            score += 1.0
        elif len(tok) >= 4 and tok[:4] in target_stems:
            score += 0.65
    return score / max(len(query_tokens), 1)


def is_guarded_query(query: str) -> bool:
    """Return True for empty, negated, interrogative, or destructive queries."""
    stripped = query.strip() if isinstance(query, str) else ""
    if not stripped or len(stripped) > 2048 or "?" in stripped:
        return True
    folded = _fold(stripped)
    return bool(
        _GUARD_NEGATION_RE.search(folded)
        or _GUARD_QUESTION_RE.search(folded)
        or _GUARD_DESTRUCTIVE_RE.search(folded)
    )


def normalize_candidates(candidates: Sequence[Mapping[str, Any]], *, max_candidates: int = MAX_CANDIDATES) -> list[dict[str, Any]]:
    """Validate and deduplicate candidate URI routes against `parse_uri`."""
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    limit = max(1, min(int(max_candidates), 64))
    for item in candidates or ():
        if not isinstance(item, Mapping):
            continue
        uri = str(item.get("uri") or "").strip()
        if not uri or uri in seen:
            continue
        parsed = parse_uri(uri)
        if not parsed.get("valid") or not parsed.get("scheme") or not parsed.get("path"):
            continue
        seen.add(uri)
        label = str(item.get("description") or item.get("desc") or item.get("label") or uri).strip()[:160]
        triggers = tuple(str(t).strip() for t in (item.get("triggers") or ()) if str(t).strip())
        normalized.append({"uri": uri, "label": label, "triggers": triggers, "parsed": parsed})
        if len(normalized) >= limit:
            break
    return normalized


def _tier1_deterministic_match(
    query: str,
    candidates: Sequence[dict[str, Any]],
    *,
    min_score: float = 0.72,
    min_margin: float = 0.20,
) -> tuple[str | None, str, float]:
    stripped = query.strip()
    by_uri = {c["uri"]: c for c in candidates}
    if stripped in by_uri:
        return stripped, "tier1_exact_uri", 1.0

    q_tokens = _tokens(stripped)
    if not q_tokens:
        return None, "none", 0.0

    scored: list[tuple[float, str]] = []
    for cand in candidates:
        uri_toks = _tokens(cand["uri"].replace("://", " ").replace("/", " "))
        label_toks = _tokens(cand["label"])
        trig_toks = _tokens(" ".join(cand["triggers"]))
        score = max(
            _stem_overlap(q_tokens, trig_toks),
            _stem_overlap(q_tokens, label_toks) * 0.95,
            _stem_overlap(q_tokens, uri_toks) * 0.90,
            _stem_overlap(q_tokens, uri_toks | label_toks | trig_toks) * 0.92,
        )
        scored.append((score, cand["uri"]))

    scored.sort(key=lambda row: (-row[0], row[1]))
    top_score, top_uri = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if top_score >= min_score and (top_score - second_score) >= min_margin:
        return top_uri, "tier1_deterministic", round(min(1.0, top_score), 4)
    return None, "ambiguous", round(top_score, 4)


def build_decision_payload(
    query: str,
    candidates: Sequence[dict[str, Any]],
    *,
    model: str = DEFAULT_DECISION_MODEL,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build a bounded `/v1/decide` payload with symmetric option labels and an option->URI map."""
    options: dict[str, str] = {}
    option_to_uri: dict[str, str] = {}
    for idx, cand in enumerate(candidates):
        key = f"op_{idx}"
        options[key] = cand["label"]
        option_to_uri[key] = cand["uri"]
    options["NO_MATCH"] = NO_MATCH_LABEL
    payload = {
        "model": model,
        "offline": True,
        "state": {"request": query.strip()},
        "questions": [
            {
                "id": "route",
                "type": "choice",
                "question": "Która trasa URI najlepiej odpowiada żądaniu?",
                "options": options,
            }
        ],
    }
    return payload, option_to_uri


def _load_decision_client(state_dir: str | Path | None = None):
    cfg_path = (
        Path(os.environ["HEYASA_AI_CONFIG"]).expanduser()
        if os.environ.get("HEYASA_AI_CONFIG")
        else Path(state_dir or os.environ.get("HEYASA_STATE_DIR") or Path.home() / ".local/share/heyasa") / "ai-runtime.json"
    )
    if not cfg_path.is_file():
        return None
    try:
        from subllm import runtime_profile
        profile = runtime_profile.ProfileStore(cfg_path).load()
        if not profile.get("enabled"):
            return None
        return runtime_profile.GatewayClient(profile, application="urirun-connector-router")
    except Exception:
        return None


def decide_uri_route(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    client: Any = None,
    state_dir: str | Path | None = None,
    max_candidates: int = MAX_CANDIDATES,
    min_deterministic_score: float = 0.72,
    min_deterministic_margin: float = 0.20,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Resolve `query` to one declared URI route using Tier 1 deterministic matching and Tier 2 `roles.decision`."""
    t0 = time.perf_counter()
    if is_guarded_query(query):
        return {
            "schema": DECISION_SCHEMA,
            "status": "no_match",
            "tier": "guard",
            "selected_uri": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    normalized = normalize_candidates(candidates, max_candidates=max_candidates)
    if not normalized:
        return {
            "schema": DECISION_SCHEMA,
            "status": "no_match",
            "tier": "empty_catalog",
            "selected_uri": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    matched_uri, tier, score = _tier1_deterministic_match(
        query,
        normalized,
        min_score=min_deterministic_score,
        min_margin=min_deterministic_margin,
    )
    if matched_uri is not None:
        return {
            "schema": DECISION_SCHEMA,
            "status": "selected",
            "tier": tier,
            "selected_uri": matched_uri,
            "score": score,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    active_client = client if client is not None else _load_decision_client(state_dir=state_dir)
    if active_client is None:
        return {
            "schema": DECISION_SCHEMA,
            "status": "unavailable",
            "tier": "tier2_unavailable",
            "selected_uri": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    profile = getattr(active_client, "profile", None)
    role_model = (
        ((profile.get("roles") or {}).get("decision") or {}).get("model")
        if isinstance(profile, Mapping)
        else None
    ) or DEFAULT_DECISION_MODEL
    payload, option_to_uri = build_decision_payload(query, normalized, model=str(role_model))
    payload["timeout"] = float(timeout_seconds)

    try:
        if callable(active_client) and not hasattr(active_client, "infer"):
            raw = active_client(payload)
        else:
            raw = active_client.infer("decision", payload)
    except Exception as exc:
        return {
            "schema": DECISION_SCHEMA,
            "status": "unavailable",
            "tier": "tier2_unavailable",
            "selected_uri": None,
            "reason": type(exc).__name__,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    answers = raw.get("answers") if isinstance(raw, Mapping) else None
    choice = answers.get("route") if isinstance(answers, Mapping) else None
    selected_uri = option_to_uri.get(choice) if isinstance(choice, str) else None
    model_used = str(raw.get("model") or role_model) if isinstance(raw, Mapping) else str(role_model)

    if selected_uri is None:
        return {
            "schema": DECISION_SCHEMA,
            "status": "no_match",
            "tier": "tier2_decision",
            "selected_uri": None,
            "model": model_used,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    return {
        "schema": DECISION_SCHEMA,
        "status": "selected",
        "tier": "tier2_decision",
        "selected_uri": selected_uri,
        "model": model_used,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }
