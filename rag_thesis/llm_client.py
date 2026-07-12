"""Dünner OpenAI-Wrapper mit Rate-Limit-Schutz, Retry und Determinismus.

Drei Schutzschichten gegen die OpenAI-TPM-Limits:

1. Token-Bucket pro Modell: wirft präventiv eine clientseitige Bremse,
   bevor das TPM-Limit überhaupt erreicht wird. Verhindert den Retry-Storm,
   den ein-Validierungslauf mal zum Hängen gebracht hat.
2. Gezielte tenacity-Retries: RateLimitError bekommt einen längeren,
   bewussteren Backoff, Transient-Fehler (InternalServerError, APITimeout)
   einen kürzeren. Stop nach 10 Versuchen statt 5.
3. Respektiere Retry-After-Header: wenn OpenAI explizit sagt, wann der
   nächste Versuch okay ist, wird genau so lange gewartet.

Per-Call-Audit: Jeder OpenAI-Aufruf (inkl. Retries) wird als eine JSON-Zeile
in outputs/0_orchestrator/api_calls.jsonl protokolliert. Die Datei wächst
über Pipeline-Läufe hinweg an und liefert die vollständige Kostenspur.

Der Client wird lazy instanziiert, damit Tests und Hilfs-Skripte 
das Modul ohne gesetzten API-Key importieren können.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from . import config

_client: Optional[OpenAI] = None
_client_lock = threading.Lock()

_LOGGER = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """Liefert einen Singleton-OpenAI-Client (lazy, threadsicher).

    Setzt einen expliziten Request-Timeout: ohne diesen wartet das SDK im
    Default ~10 Minuten auf eine Antwort. Bei OpenAI-seitigen Ausfällen
    (intermittierende 500er, Slow-API-Phasen) bedeutete das vor diesem Fix,
    dass die Pipeline pro Call eine Minute hängt, bevor tenacity überhaupt
    einen Retry auslöst. 60s sind ein guter Kompromiss zwischen
    "echten" langsamen Calls (Vision/Structured Output) und schnellem
    Failover bei Server-Hängern.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not config.OPENAI_API_KEY:
                    raise RuntimeError(
                        "OPENAI_API_KEY ist nicht gesetzt. "
                        "Bitte .env mit gültigem Key anlegen (siehe .env.example)."
                    )
                _client = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    timeout=config.OPENAI_REQUEST_TIMEOUT,
                    max_retries=0,  # Retries macht tenacity, nicht das SDK.
                )
    return _client


# ─── Token-Bucket: clientseitiger TPM-Schutz ──────────────────────────────────

class _TokenBucket:
    """Schiebefenster-Token-Bucket über 60-Sekunden-TPM.

    reserve(tokens) blockiert genau so lange, bis tokens zusätzlich
    in das 60s-Fenster passen. Damit wird das OpenAI-Limit nie
    überschritten. Retry-Storms werden strukturell unmöglich.
    """

    def __init__(self, tpm_limit: int):
        self.limit = tpm_limit
        self._events: Deque[Tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def reserve(self, tokens: int) -> None:
        # Sehr große Einzel-Reservierungen würden ewig blocken, also kappen.
        tokens = min(tokens, max(self.limit - 1, 1))
        with self._cond:
            while True:
                now = time.monotonic()
                while self._events and self._events[0][0] <= now - 60:
                    self._events.popleft()
                in_window = sum(t for _, t in self._events)
                if in_window + tokens <= self.limit:
                    self._events.append((now, tokens))
                    return
                wait_for = 60 - (now - self._events[0][0]) + 0.05
                self._cond.wait(timeout=max(wait_for, 0.05))


# Token-Buckets nur für die teuren Modelle, weil mini/Embedding selten der Engpass sind.
_buckets: Dict[str, _TokenBucket] = {}
_buckets_lock = threading.Lock()


def _bucket_for(model: str) -> Optional[_TokenBucket]:
    """Liefert (oder erzeugt) den Token-Bucket für ein Modell.

    Aktiv nur für gpt-4o-Varianten (alles, was das Judge-/Advanced-Modell
    sein könnte). Dort ist das TPM-Limit der dominante Engpass.
    """
    if "mini" in model or "embedding" in model.lower():
        return None
    with _buckets_lock:
        if model not in _buckets:
            _buckets[model] = _TokenBucket(config.JUDGE_TPM_LIMIT)
        return _buckets[model]


def _estimate_tokens(messages: Sequence[dict], max_tokens: int) -> int:
    """Grobe Token-Schätzung: 4 Zeichen ≈ 1 Token (empirisch für DE/EN).

    Der Schätzer muss eher überschätzen: Bei Unterschätzung würde der Bucket 
    reißen und 429er auslösen. Aufgerundet wird auf 1.2×, plus Output-Budget.
    """
    chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chars += len(part["text"])
    return int(chars / 4 * 1.2) + max_tokens


# ─── Retry-Policy ─────────────────────────────────────────────────────────────

# Zwei Retry-Policies: längeres Backoff für RateLimits, kürzeres für transiente Fehler.
_RETRY_RATELIMIT = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(min=4, max=120),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
    reraise=True,
)

_RETRY_TRANSIENT = retry(
    retry=retry_if_exception_type((InternalServerError, APITimeoutError, APIConnectionError)),
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(_LOGGER, logging.WARNING),
    reraise=True,
)


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _sleep_from_retry_after(exc: RateLimitError) -> bool:
    """Wenn OpenAI im 429-Body sagt "try again in Xs", schlafe genau so lange.

    Liefert True, wenn ein Retry-Hint extrahiert und respektiert wurde.
    """
    msg = str(getattr(exc, "message", "") or exc)
    m = _RETRY_AFTER_RE.search(msg)
    if not m:
        return False
    try:
        wait = float(m.group(1)) + 0.1
    except ValueError:
        return False
    if wait > 0:
        _LOGGER.info(f"Respektiere Retry-After-Hint von OpenAI: schlafe {wait:.2f}s.")
        time.sleep(min(wait, 60.0))
        return True
    return False


# ─── Per-Call-Logging (JSONL) ────────────────────────────────────────────────
#
# Jeder OpenAI-Call wird als eine Zeile in outputs/0_orchestrator/api_calls.jsonl
# protokolliert. Eine vollständige Kostenspur über alle Pipeline-Läufe hinweg.
# Felder: ts, kind, model, status, latency_s, prompt_tokens, completion_tokens,
# total_tokens, cached_tokens (durch Prompt-Caching vergünstigte Input-Tokens),
# n_messages (chat) bzw. n_inputs (embed), error_type (nur bei Fehler).

_log_lock = threading.Lock()
_log_path_resolved: Optional[Any] = None


def _api_log_path() -> Any:
    """Liefere den Log-Pfad lazy. config wird erst beim ersten Call benötigt."""
    global _log_path_resolved
    if _log_path_resolved is None:
        config.DIR_ORCHESTRATOR.mkdir(parents=True, exist_ok=True)
        _log_path_resolved = config.DIR_ORCHESTRATOR / "api_calls.jsonl"
    return _log_path_resolved


def _api_log(record: Dict[str, Any]) -> None:
    """Hänge eine JSON-Zeile thread-safe an die API-Log-Datei an.

    Schwerwiegende Fehler beim Loggen werden geschluckt.
    Das Logging darf den eigentlichen Pipeline-Lauf nicht abbrechen.
    """
    record["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    try:
        with _log_lock:
            with open(_api_log_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        _LOGGER.debug(f"API-Log-Schreiben fehlgeschlagen: {exc}")


def _usage_dict(resp: Any) -> Dict[str, int]:
    """Extrahiere prompt/completion/total Tokens aus der OpenAI-Response (falls vorhanden).

    Erfasst zusätzlich cached_tokens aus prompt_tokens_details: das sind die
    Input-Tokens, die OpenAIs automatisches Prompt-Caching vergünstigt abrechnet. 
    So lässt sich im Per-Call-Audit auswerten, wie viel tatsächlich
    gecacht wurde (z.B. bei wiederholten System-Prompts in Schritt 2 und 5).
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    if cached is None and isinstance(details, dict):
        cached = details.get("cached_tokens", 0)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cached_tokens": int(cached or 0),
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def chat_complete(
    messages: Sequence[dict],
    model: str,
    *,
    max_tokens: int,
    temperature: float = config.TEMPERATURE,
    seed: Optional[int] = config.RANDOM_SEED,
) -> str:
    """Chat-Completion mit Token-Bucket + Retry. Liefert den Text-Content."""
    bucket = _bucket_for(model)
    if bucket is not None:
        bucket.reserve(_estimate_tokens(messages, max_tokens))
    return _do_chat(messages, model, max_tokens=max_tokens,
                     temperature=temperature, seed=seed)


@_RETRY_TRANSIENT
@_RETRY_RATELIMIT
def _do_chat(
    messages: Sequence[dict],
    model: str,
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
) -> str:
    # Jeder Versuch (inkl. Retries) wird einzeln geloggt -> exakte Call-Zählung.
    t0 = time.monotonic()
    try:
        resp = get_client().chat.completions.create(
            model=model,
            messages=cast(Any, list(messages)),  # openai erwartet typisierte Message-Dicts. Laufzeit-Dicts sind korrekt
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        _api_log({
            "kind": "chat", "model": model, "status": "ok",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_messages": len(messages),
            **_usage_dict(resp),
        })
        return resp.choices[0].message.content or ""
    except RateLimitError as exc:
        _api_log({
            "kind": "chat", "model": model, "status": "rate_limit",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_messages": len(messages), "error_type": type(exc).__name__,
        })
        _sleep_from_retry_after(exc)
        raise
    except Exception as exc:
        _api_log({
            "kind": "chat", "model": model, "status": "error",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_messages": len(messages), "error_type": type(exc).__name__,
        })
        raise


def structured_complete(
    messages: Sequence[dict],
    model: str,
    response_format: Any,
    *,
    max_tokens: int,
    temperature: float = config.TEMPERATURE,
    seed: Optional[int] = config.RANDOM_SEED,
) -> Any:
    """Structured-Output-Completion mit Token-Bucket + Retry."""
    bucket = _bucket_for(model)
    if bucket is not None:
        bucket.reserve(_estimate_tokens(messages, max_tokens))
    return _do_structured(messages, model, response_format,
                           max_tokens=max_tokens, temperature=temperature, seed=seed)


@_RETRY_TRANSIENT
@_RETRY_RATELIMIT
def _do_structured(
    messages: Sequence[dict],
    model: str,
    response_format: Any,
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
) -> Any:
    t0 = time.monotonic()
    try:
        resp = get_client().beta.chat.completions.parse(
            model=model,
            messages=cast(Any, list(messages)),  # openai erwartet typisierte Message-Dicts. Laufzeit-Dicts sind korrekt
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        _api_log({
            "kind": "structured", "model": model, "status": "ok",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_messages": len(messages),
            **_usage_dict(resp),
        })
        return resp
    except RateLimitError as exc:
        _api_log({
            "kind": "structured", "model": model, "status": "rate_limit",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_messages": len(messages), "error_type": type(exc).__name__,
        })
        _sleep_from_retry_after(exc)
        raise
    except Exception as exc:
        _api_log({
            "kind": "structured", "model": model, "status": "error",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_messages": len(messages), "error_type": type(exc).__name__,
        })
        raise


@_RETRY_TRANSIENT
@_RETRY_RATELIMIT
def embed_texts(texts: Sequence[str], *, model: str = config.EMBEDDING_MODEL) -> List[List[float]]:
    """Embedding-Wrapper mit Retry. Kein Token-Bucket. Embedding-Limits sind hoch."""
    t0 = time.monotonic()
    try:
        resp = get_client().embeddings.create(input=list(texts), model=model)
        _api_log({
            "kind": "embed", "model": model, "status": "ok",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_inputs": len(texts),
            **_usage_dict(resp),
        })
        return [list(map(float, e.embedding)) for e in resp.data]
    except RateLimitError as exc:
        _api_log({
            "kind": "embed", "model": model, "status": "rate_limit",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_inputs": len(texts), "error_type": type(exc).__name__,
        })
        _sleep_from_retry_after(exc)
        raise
    except Exception as exc:
        _api_log({
            "kind": "embed", "model": model, "status": "error",
            "latency_s": round(time.monotonic() - t0, 3),
            "n_inputs": len(texts), "error_type": type(exc).__name__,
        })
        raise
