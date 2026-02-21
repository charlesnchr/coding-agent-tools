#!/usr/bin/env python3
"""Unified token and estimated cost usage across coding agents."""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


SOURCE_ALIASES = {
    "claude": "claude",
    "cc": "claude",
    "codex": "codex",
    "cx": "codex",
    "openclaw": "openclaw",
    "oc": "openclaw",
    "claw": "openclaw",
    "opencode": "opencode",
    "oe": "opencode",
    "code": "opencode",
    "openwhispr": "openwhispr",
    "ow": "openwhispr",
    "whispr": "openwhispr",
    "whisper": "openwhispr",
}

GROUP_MODES = {"daily", "weekly", "monthly"}
LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)


@dataclass
class ModelBucket:
    provider: str = ""
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total: int = 0
    cost: float = 0.0


class PricingResolver:
    def __init__(self) -> None:
        self._prices = self._fetch_litellm_prices()
        self._lookup_cache: dict[tuple[str, str], Optional[dict]] = {}
        self._model_aliases = {
            # Codex aliases
            "gpt-5.3-codex": "gpt-5.2-codex",
            # Custom OpenCode routes seen in local setup
            "antigravity-gemini-3.1-pro": "gemini-2.5-pro",
            "gemini-3.1-pro-preview": "gemini-2.5-pro",
            "gemini-3.1-pro-preview-customtools": "gemini-2.5-pro",
            "antigravity-claude-opus-4-6-thinking": "claude-opus-4-1",
        }

    def estimate_cost(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
    ) -> Optional[float]:
        rates = self._find_rates(model, provider)
        if not rates:
            return None

        input_rate = rates.get("input", 0.0)
        output_rate = rates.get("output", 0.0)
        cache_read_rate = rates.get("cache_read")
        cache_write_rate = rates.get("cache_write")

        if cache_read_rate is None:
            cache_read_rate = input_rate
        if cache_write_rate is None:
            cache_write_rate = input_rate

        return (
            input_tokens * input_rate
            + output_tokens * output_rate
            + cache_read_tokens * cache_read_rate
            + cache_write_tokens * cache_write_rate
        )

    def _find_rates(self, model: str, provider: str) -> Optional[dict]:
        key = (provider or "", model or "")
        if key in self._lookup_cache:
            return self._lookup_cache[key]

        candidates = self._candidate_keys(model, provider)
        for candidate in candidates:
            rates = self._extract_rates(self._prices.get(candidate))
            if rates:
                self._lookup_cache[key] = rates
                return rates

        # Secondary lookup: model suffix / substring match
        lowered = [c.lower() for c in candidates]
        for price_key, value in self._prices.items():
            price_key_lower = price_key.lower()
            if any(
                price_key_lower == c
                or price_key_lower.endswith("/" + c)
                or c in price_key_lower
                for c in lowered
            ):
                rates = self._extract_rates(value)
                if rates:
                    self._lookup_cache[key] = rates
                    return rates

        self._lookup_cache[key] = None
        return None

    def _candidate_keys(self, model: str, provider: str) -> list[str]:
        model = (model or "").strip()
        provider = (provider or "").strip()
        raw = [model]

        if model in self._model_aliases:
            raw.append(self._model_aliases[model])

        # Strip common custom prefixes progressively.
        stripped = model
        for prefix in ("antigravity-", "openai/", "azure/", "google/"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                raw.append(stripped)

        if "-preview-customtools" in stripped:
            raw.append(stripped.replace("-preview-customtools", ""))
        if "-preview" in stripped:
            raw.append(stripped.replace("-preview", ""))

        out = []
        seen = set()
        for item in raw:
            if not item:
                continue
            for candidate in (
                item,
                item.lower(),
                f"{provider}/{item}" if provider else "",
                f"{provider}/{item.lower()}" if provider else "",
                f"openrouter/{provider}/{item}" if provider else "",
            ):
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    out.append(candidate)
        return out

    @staticmethod
    def _extract_rates(data: Optional[dict]) -> Optional[dict]:
        if not isinstance(data, dict):
            return None
        if "input_cost_per_token" not in data or "output_cost_per_token" not in data:
            return None
        return {
            "input": float(data.get("input_cost_per_token") or 0.0),
            "output": float(data.get("output_cost_per_token") or 0.0),
            "cache_write": data.get("cache_creation_input_token_cost"),
            "cache_read": data.get("cache_read_input_token_cost"),
        }

    @staticmethod
    def _fetch_litellm_prices() -> dict:
        try:
            with urllib.request.urlopen(LITELLM_URL, timeout=15) as response:
                data = json.loads(response.read())
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def to_period(ts: datetime, mode: str) -> str:
    if mode == "daily":
        return ts.strftime("%Y-%m-%d")
    if mode == "monthly":
        return ts.strftime("%Y-%m")
    year, week, _ = ts.isocalendar()
    return f"{year}-W{week:02d}"


def should_show(source_name: str, filter_set: set[str]) -> bool:
    return not filter_set or source_name in filter_set


def add_usage(
    store: dict[str, dict[str, ModelBucket]],
    totals: ModelBucket,
    period: str,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    total_tokens: int,
    cost_usd: float,
) -> None:
    if period not in store:
        store[period] = {}
    if model not in store[period]:
        store[period][model] = ModelBucket(provider=provider)
    bucket = store[period][model]

    bucket.input += input_tokens
    bucket.output += output_tokens
    bucket.cache_read += cache_read_tokens
    bucket.cache_write += cache_write_tokens
    bucket.total += total_tokens
    bucket.cost += cost_usd

    totals.input += input_tokens
    totals.output += output_tokens
    totals.cache_read += cache_read_tokens
    totals.cache_write += cache_write_tokens
    totals.total += total_tokens
    totals.cost += cost_usd


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_cost(value: float) -> str:
    return f"${value:,.2f}"


def model_label(model: str, provider: str) -> str:
    if provider:
        return f"{model} ({provider})"
    return model


def print_table(
    title: str,
    usage_by_period: dict[str, dict[str, ModelBucket]],
    totals: ModelBucket,
    mode: str,
    breakdown: bool,
    col_labels: Optional[dict[str, str]] = None,
    input_header: str = "Input",
) -> None:
    if not usage_by_period:
        print(f"\n{title}\nNo data found.")
        return

    entries = []
    for period in sorted(usage_by_period.keys()):
        parts = period.split("-")
        year = parts[0]
        period_tail = ""
        if mode == "daily":
            period_tail = f"{parts[1]}-{parts[2]}" if len(parts) == 3 else ""
        elif mode in ("monthly", "weekly"):
            period_tail = parts[1] if len(parts) > 1 else ""

        models = sorted(usage_by_period[period].keys())

        if not breakdown and len(models) > 1:
            agg = ModelBucket()
            for name in models:
                item = usage_by_period[period][name]
                agg.input += item.input
                agg.output += item.output
                agg.cache_read += item.cache_read
                agg.cache_write += item.cache_write
                agg.total += item.total
                agg.cost += item.cost
            entries.append(
                {
                    "type": "data",
                    "date": year,
                    "model": f"- Multiple ({len(models)})",
                    "input": fmt_int(agg.input),
                    "output": fmt_int(agg.output),
                    "cache_w": fmt_int(agg.cache_write),
                    "cache_r": fmt_int(agg.cache_read),
                    "total": fmt_int(agg.total),
                    "cost": fmt_cost(agg.cost),
                }
            )
            if period_tail:
                entries.append({"type": "date_cont", "date": period_tail})
            entries.append({"type": "sep"})
            continue

        if breakdown and len(models) > 1:
            agg = ModelBucket()
            for name in models:
                item = usage_by_period[period][name]
                agg.input += item.input
                agg.output += item.output
                agg.cache_read += item.cache_read
                agg.cache_write += item.cache_write
                agg.total += item.total
                agg.cost += item.cost
            entries.append(
                {
                    "type": "data",
                    "date": year,
                    "model": "",
                    "input": fmt_int(agg.input),
                    "output": fmt_int(agg.output),
                    "cache_w": fmt_int(agg.cache_write),
                    "cache_r": fmt_int(agg.cache_read),
                    "total": fmt_int(agg.total),
                    "cost": fmt_cost(agg.cost),
                }
            )
            if period_tail:
                entries.append({"type": "date_cont", "date": period_tail})
            entries.append({"type": "sep"})
            for idx, name in enumerate(models):
                item = usage_by_period[period][name]
                marker = "└─" if idx == len(models) - 1 else "├─"
                entries.append(
                    {
                        "type": "data",
                        "date": f"  {marker}",
                        "model": model_label(name, item.provider),
                        "input": fmt_int(item.input),
                        "output": fmt_int(item.output),
                        "cache_w": fmt_int(item.cache_write),
                        "cache_r": fmt_int(item.cache_read),
                        "total": fmt_int(item.total),
                        "cost": fmt_cost(item.cost),
                    }
                )
            entries.append({"type": "sep"})
            continue

        model_name = models[0]
        item = usage_by_period[period][model_name]
        entries.append(
            {
                "type": "data",
                "date": year,
                "model": f"- {model_label(model_name, item.provider)}",
                "input": fmt_int(item.input),
                "output": fmt_int(item.output),
                "cache_w": fmt_int(item.cache_write),
                "cache_r": fmt_int(item.cache_read),
                "total": fmt_int(item.total),
                "cost": fmt_cost(item.cost),
            }
        )
        if period_tail:
            entries.append({"type": "date_cont", "date": period_tail})
        entries.append({"type": "sep"})

    total_row = {
        "date": "Total",
        "model": "",
        "input": fmt_int(totals.input),
        "output": fmt_int(totals.output),
        "cache_w": fmt_int(totals.cache_write),
        "cache_r": fmt_int(totals.cache_read),
        "total": fmt_int(totals.total),
        "cost": fmt_cost(totals.cost),
    }

    all_data = [e for e in entries if e["type"] == "data"] + [total_row]
    all_dates = [e for e in entries if e["type"] in ("data", "date_cont")]

    dw = max(9, max((len(e["date"]) for e in all_dates), default=9))
    mw = max(20, max((len(e["model"]) for e in all_data), default=20))
    iw = max(len(input_header), max(len(e["input"]) for e in all_data))
    ow = max(len("Output"), max(len(e["output"]) for e in all_data))
    cww = max(len("Create"), max(len(e["cache_w"]) for e in all_data))
    crw = max(len("Read"), max(len(e["cache_r"]) for e in all_data))
    tw = max(len("Tokens"), max(len(e["total"]) for e in all_data))
    cow = max(len("(USD)"), max(len(e["cost"]) for e in all_data))

    use_color = sys.stdout.isatty() and not os.getenv("NO_COLOR")
    dim = "\033[90m" if use_color else ""
    cyan = "\033[36m" if use_color else ""
    reset = "\033[39m" if use_color else ""

    def hl(l: str, m: str, r: str) -> str:
        segs = [f"{'─' * (w + 2)}" for w in (dw, mw, iw, ow, cww, crw, tw, cow)]
        return f"{dim}{l}{m.join(segs)}{r}{reset}"

    def rl(
        d: str,
        m: str,
        i: str,
        o: str,
        cw: str,
        cr: str,
        t: str,
        c: str,
        color: str = "",
    ) -> str:
        ec = reset if color else ""
        return (
            f"{dim}│{reset}{color} {d.ljust(dw)} {ec}"
            f"{dim}│{reset}{color} {m.ljust(mw)} {ec}"
            f"{dim}│{reset}{color} {i.rjust(iw)} {ec}"
            f"{dim}│{reset}{color} {o.rjust(ow)} {ec}"
            f"{dim}│{reset}{color} {cw.rjust(cww)} {ec}"
            f"{dim}│{reset}{color} {cr.rjust(crw)} {ec}"
            f"{dim}│{reset}{color} {t.rjust(tw)} {ec}"
            f"{dim}│{reset}{color} {c.rjust(cow)} {ec}{dim}│{reset}"
        )

    col1 = {"weekly": "Week", "monthly": "Month"}.get(mode, "Date")
    labels = col_labels or {}
    h_input = labels.get("input", input_header)
    h_col5 = labels.get("col5", "Cache")
    h_col5_sub = labels.get("col5_sub", "Create")
    h_col6 = labels.get("col6", "Cache")
    h_col6_sub = labels.get("col6_sub", "Read")

    print(f"\n{title}")
    print(f"\n{hl('┌', '┬', '┐')}")
    print(rl(col1, "Models", h_input, "Output", h_col5, h_col6, "Total", "Cost", cyan))
    print(rl("", "", "", "", h_col5_sub, h_col6_sub, "Tokens", "(USD)", cyan))
    print(hl("├", "┼", "┤"))

    for e in entries:
        if e["type"] == "sep":
            print(hl("├", "┼", "┤"))
        elif e["type"] == "date_cont":
            print(rl(e["date"], "", "", "", "", "", "", ""))
        elif e["type"] == "data":
            print(
                rl(
                    e["date"],
                    e["model"],
                    e["input"],
                    e["output"],
                    e["cache_w"],
                    e["cache_r"],
                    e["total"],
                    e["cost"],
                )
            )

    print(
        rl(
            total_row["date"],
            total_row["model"],
            total_row["input"],
            total_row["output"],
            total_row["cache_w"],
            total_row["cache_r"],
            total_row["total"],
            total_row["cost"],
        )
    )
    print(hl("└", "┴", "┘"))


def collect_claude(
    mode: str,
    pricing: PricingResolver,
) -> tuple[dict[str, dict[str, ModelBucket]], ModelBucket]:
    store: dict[str, dict[str, ModelBucket]] = {}
    totals = ModelBucket()
    seen_hashes = set()

    pattern = os.path.expanduser("~/.claude/projects/*/*.jsonl")
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") != "assistant":
                        continue
                    message = obj.get("message", {})
                    usage = message.get("usage", {})
                    input_tokens = int(usage.get("input_tokens", 0) or 0)
                    output_tokens = int(usage.get("output_tokens", 0) or 0)
                    cache_write_tokens = int(
                        usage.get("cache_creation_input_tokens", 0) or 0
                    )
                    cache_read_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
                    if input_tokens + output_tokens + cache_write_tokens + cache_read_tokens <= 0:
                        continue

                    message_id = message.get("id", "")
                    request_id = obj.get("requestId", "")
                    if message_id and request_id:
                        dedupe_key = f"{message_id}:{request_id}"
                        if dedupe_key in seen_hashes:
                            continue
                        seen_hashes.add(dedupe_key)

                    model = str(message.get("model", "unknown"))
                    provider = "anthropic"
                    total_tokens = int(
                        usage.get("total_tokens", 0)
                        or input_tokens
                        + output_tokens
                        + cache_write_tokens
                        + cache_read_tokens
                    )

                    ts = obj.get("timestamp")
                    if not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                    estimated_cost = pricing.estimate_cost(
                        model,
                        provider,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                    )
                    cost = float(estimated_cost or 0.0)

                    add_usage(
                        store,
                        totals,
                        to_period(dt, mode),
                        model,
                        provider,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                        total_tokens,
                        cost,
                    )
        except OSError:
            continue

    return store, totals


def collect_openclaw(
    mode: str,
    pricing: PricingResolver,
) -> tuple[dict[str, dict[str, ModelBucket]], ModelBucket]:
    store: dict[str, dict[str, ModelBucket]] = {}
    totals = ModelBucket()

    pattern = os.path.expanduser("~/.openclaw/agents/*/sessions/*.jsonl")
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") != "message":
                        continue
                    message = obj.get("message", {})
                    if message.get("role") != "assistant":
                        continue
                    usage = message.get("usage", {})

                    input_tokens = int(usage.get("input", 0) or 0)
                    output_tokens = int(usage.get("output", 0) or 0)
                    cache_read_tokens = int(usage.get("cacheRead", 0) or 0)
                    cache_write_tokens = int(usage.get("cacheWrite", 0) or 0)
                    total_tokens = int(usage.get("totalTokens", 0) or 0)
                    if total_tokens <= 0:
                        continue

                    model = str(message.get("model", "unknown"))
                    provider = str(message.get("provider", ""))

                    raw_cost = usage.get("cost", {})
                    if isinstance(raw_cost, dict):
                        observed_cost = float(raw_cost.get("total", 0.0) or 0.0)
                    else:
                        observed_cost = float(raw_cost or 0.0)

                    estimated_cost = pricing.estimate_cost(
                        model,
                        provider,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                    )
                    cost = float(estimated_cost if estimated_cost is not None else observed_cost)

                    ts = obj.get("timestamp")
                    if not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                    add_usage(
                        store,
                        totals,
                        to_period(dt, mode),
                        model,
                        provider,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                        total_tokens,
                        cost,
                    )
        except OSError:
            continue

    return store, totals


def collect_opencode(
    mode: str,
    pricing: PricingResolver,
) -> tuple[dict[str, dict[str, ModelBucket]], ModelBucket]:
    store: dict[str, dict[str, ModelBucket]] = {}
    totals = ModelBucket()

    db_path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    if not db_path.exists():
        return store, totals

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                time_created,
                json_extract(data, '$.modelID') as model,
                json_extract(data, '$.providerID') as provider,
                json_extract(data, '$.tokens.input') as input_tokens,
                json_extract(data, '$.tokens.output') as output_tokens,
                json_extract(data, '$.tokens.cache.read') as cache_read_tokens,
                json_extract(data, '$.tokens.cache.write') as cache_write_tokens,
                json_extract(data, '$.tokens.total') as total_tokens,
                json_extract(data, '$.cost') as observed_cost
            FROM message
            WHERE json_extract(data, '$.role') = 'assistant'
              AND json_extract(data, '$.tokens.total') > 0
            ORDER BY time_created ASC
            """
        )
        for row in cursor.fetchall():
            (
                ts_ms,
                model,
                provider,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                total_tokens,
                observed_cost,
            ) = row

            model = str(model or "unknown")
            provider = str(provider or "")
            input_tokens = int(input_tokens or 0)
            output_tokens = int(output_tokens or 0)
            cache_read_tokens = int(cache_read_tokens or 0)
            cache_write_tokens = int(cache_write_tokens or 0)
            total_tokens = int(total_tokens or 0)
            observed_cost = float(observed_cost or 0.0)

            estimated_cost = pricing.estimate_cost(
                model,
                provider,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
            )
            cost = float(estimated_cost if estimated_cost is not None else observed_cost)

            dt = datetime.fromtimestamp(float(ts_ms) / 1000.0)

            add_usage(
                store,
                totals,
                to_period(dt, mode),
                model,
                provider,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                total_tokens,
                cost,
            )

        conn.close()
    except sqlite3.Error as exc:
        print(f"OpenCode DB error: {exc}", file=sys.stderr)

    return store, totals


def collect_codex(
    mode: str,
    pricing: PricingResolver,
) -> tuple[dict[str, dict[str, ModelBucket]], ModelBucket]:
    store: dict[str, dict[str, ModelBucket]] = {}
    totals = ModelBucket()

    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return store, totals

    session_files = list(sessions_root.glob("**/rollout-*.jsonl"))
    for file_path in session_files:
        current_model = "unknown"
        current_provider = "openai"
        current_turn_id = ""
        session_id = file_path.stem
        seen_turns = set()

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    obj_type = obj.get("type")
                    payload = obj.get("payload", {})

                    if obj_type == "session_meta":
                        session_id = payload.get("id", session_id)
                        current_provider = payload.get("model_provider", "openai")
                        continue

                    if obj_type == "turn_context":
                        current_model = payload.get("model", current_model)
                        current_turn_id = payload.get("turn_id", "")
                        continue

                    if obj_type != "event_msg" or payload.get("type") != "token_count":
                        continue

                    info = payload.get("info")
                    if not isinstance(info, dict):
                        continue
                    usage = info.get("last_token_usage")
                    if not isinstance(usage, dict):
                        continue

                    turn_key = current_turn_id or payload.get("turn_id", "")
                    if turn_key:
                        dedupe = f"{session_id}:{turn_key}"
                        if dedupe in seen_turns:
                            continue
                        seen_turns.add(dedupe)

                    input_tokens = int(usage.get("input_tokens", 0) or 0)
                    output_tokens = int(usage.get("output_tokens", 0) or 0)
                    cache_read_tokens = int(usage.get("cached_input_tokens", 0) or 0)
                    cache_write_tokens = 0
                    total_tokens = int(
                        usage.get("total_tokens", 0)
                        or input_tokens
                        + output_tokens
                        + cache_read_tokens
                    )
                    if total_tokens <= 0:
                        continue

                    ts = obj.get("timestamp")
                    if not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        continue

                    estimated_cost = pricing.estimate_cost(
                        current_model,
                        current_provider,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                    )
                    cost = float(estimated_cost or 0.0)

                    add_usage(
                        store,
                        totals,
                        to_period(dt, mode),
                        current_model,
                        current_provider,
                        input_tokens,
                        output_tokens,
                        cache_read_tokens,
                        cache_write_tokens,
                        total_tokens,
                        cost,
                    )
        except OSError:
            continue

    return store, totals


def collect_openwhispr(
    mode: str,
) -> tuple[dict[str, dict[str, ModelBucket]], ModelBucket]:
    store: dict[str, dict[str, ModelBucket]] = {}
    totals = ModelBucket()

    db_path = Path.home() / "Library" / "Application Support" / "open-whispr" / "transcriptions.db"
    if not db_path.exists():
        return store, totals

    token_pricing = {
        "gpt-4o-transcribe": {"audio_input": 6.00, "output": 10.00},
        "gpt-4o-mini-transcribe": {"audio_input": 3.00, "output": 5.00},
        "whisper-1": {"audio_input": 6.00, "output": 0.00},
    }
    hourly_pricing = {
        "whisper-large-v3": 0.111,
        "whisper-large-v3-turbo": 0.04,
    }

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transcription_usage'"
        )
        if not cursor.fetchone():
            conn.close()
            return store, totals

        cursor.execute("PRAGMA table_info(transcription_usage)")
        columns = [row[1] for row in cursor.fetchall()]
        has_duration = "duration_seconds" in columns

        if has_duration:
            cursor.execute(
                """
                SELECT
                    created_at,
                    model,
                    provider,
                    input_tokens,
                    output_tokens,
                    audio_tokens,
                    total_tokens,
                    duration_seconds
                FROM transcription_usage
                ORDER BY created_at ASC
                """
            )
        else:
            cursor.execute(
                """
                SELECT
                    created_at,
                    model,
                    provider,
                    input_tokens,
                    output_tokens,
                    audio_tokens,
                    total_tokens,
                    0 as duration_seconds
                FROM transcription_usage
                ORDER BY created_at ASC
                """
            )

        for row in cursor.fetchall():
            (
                ts_str,
                model,
                provider,
                _input_tokens,
                output_tokens,
                audio_tokens,
                total_tokens,
                duration_seconds,
            ) = row

            model = str(model or "unknown")
            provider = str(provider or "")
            output_tokens = int(output_tokens or 0)
            audio_tokens = int(audio_tokens or 0)
            total_tokens = int(total_tokens or 0)
            duration_seconds = float(duration_seconds or 0.0)

            try:
                dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            except ValueError:
                try:
                    dt = datetime.strptime(str(ts_str), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue

            hourly = hourly_pricing.get(model)
            if hourly and duration_seconds > 0:
                billed_seconds = max(duration_seconds, 10.0)
                cost = (billed_seconds / 3600.0) * hourly
                input_value = int(round(duration_seconds))
            else:
                rate = token_pricing.get(model, {"audio_input": 6.0, "output": 10.0})
                cost = (
                    audio_tokens * float(rate["audio_input"])
                    + output_tokens * float(rate["output"])
                ) / 1_000_000.0
                input_value = audio_tokens

            add_usage(
                store,
                totals,
                to_period(dt, mode),
                model,
                provider,
                input_value,
                output_tokens,
                0,
                0,
                total_tokens,
                float(cost),
            )

        conn.close()
    except sqlite3.Error as exc:
        print(f"OpenWhispr DB error: {exc}", file=sys.stderr)

    return store, totals


def parse_args(argv: list[str]) -> tuple[str, bool, set[str]]:
    mode = "daily"
    breakdown = False
    source_filter: set[str] = set()

    for arg in argv:
        lower = arg.lower()
        if lower in GROUP_MODES:
            mode = lower
            continue
        if lower in ("-b", "--breakdown"):
            breakdown = True
            continue
        if lower in ("-h", "--help"):
            print_help_and_exit(0)
        if lower in SOURCE_ALIASES:
            source_filter.add(SOURCE_ALIASES[lower])
            continue
        print(f"Unknown argument: {arg}", file=sys.stderr)
        print_help_and_exit(1)

    return mode, breakdown, source_filter


def print_help_and_exit(code: int) -> None:
    print(
        """agent-usage - Unified usage across coding agents

Usage:
  agent-usage [daily|weekly|monthly] [--breakdown] [sources...]

Sources:
  claude|cc, codex|cx, openclaw|oc|claw, opencode|oe|code, openwhispr|ow|whispr

Examples:
  agent-usage
  agent-usage weekly --breakdown
  agent-usage monthly opencode codex
"""
    )
    raise SystemExit(code)


def main() -> None:
    mode, breakdown, source_filter = parse_args(sys.argv[1:])
    pricing = PricingResolver()

    if should_show("claude", source_filter):
        usage, totals = collect_claude(mode, pricing)
        print_table("Claude Code Usage", usage, totals, mode, breakdown)

    if should_show("codex", source_filter):
        usage, totals = collect_codex(mode, pricing)
        print_table("Codex Usage", usage, totals, mode, breakdown)

    if should_show("openclaw", source_filter):
        usage, totals = collect_openclaw(mode, pricing)
        print_table("OpenClaw Usage", usage, totals, mode, breakdown)

    if should_show("opencode", source_filter):
        usage, totals = collect_opencode(mode, pricing)
        print_table("OpenCode Usage", usage, totals, mode, breakdown)

    if should_show("openwhispr", source_filter):
        usage, totals = collect_openwhispr(mode)
        print_table(
            "OpenWhispr Usage",
            usage,
            totals,
            mode,
            breakdown,
            col_labels={
                "col5": "",
                "col5_sub": "",
                "col6": "",
                "col6_sub": "",
            },
            input_header="Audio",
        )


if __name__ == "__main__":
    main()
