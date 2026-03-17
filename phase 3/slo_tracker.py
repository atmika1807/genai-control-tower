"""
SLO tracker
Runs as a background process, polling MLflow every N minutes
and alerting when SLO targets are breached.

SLO targets (configurable):
  - p95 query latency  < 200ms
  - SLO compliance     ≥ 95%
  - Avg rerank score   ≥ 0.70
  - Daily embed cost   < $1.00

Alerts are written to stdout (structured log) and optionally to a
webhook (Slack / PagerDuty / custom).
"""

from __future__ import annotations

import asyncio
import httpx
import structlog
from dataclasses import dataclass
from typing import Callable

log = structlog.get_logger(__name__)


@dataclass
class SLOTarget:
    name: str
    metric: str
    threshold: float
    operator: str   # "lt" | "gt" | "gte" | "lte"
    severity: str   # "warn" | "critical"


DEFAULT_SLOS: list[SLOTarget] = [
    SLOTarget("p95 latency",     "p95_latency_ms",     200.0,  "lt",  "critical"),
    SLOTarget("SLO compliance",  "slo_compliance_pct",  95.0,  "gte", "warn"),
    SLOTarget("rerank quality",  "avg_rerank_score",     0.70,  "gte", "warn"),
]


class SLOTracker:

    def __init__(
        self,
        dashboard_api_url: str = "http://localhost:8002",
        poll_interval_s: int   = 300,        # 5 min
        webhook_url: str | None = None,
        slos: list[SLOTarget]   = DEFAULT_SLOS,
    ):
        self._api            = dashboard_api_url
        self._interval       = poll_interval_s
        self._webhook_url    = webhook_url
        self._slos           = slos
        self._breach_state:  dict[str, bool] = {}   # metric → currently breaching

    async def run_forever(self) -> None:
        log.info("slo_tracker_started", interval_s=self._interval)
        while True:
            await self._check()
            await asyncio.sleep(self._interval)

    async def _check(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self._api}/metrics/query")
                r.raise_for_status()
                metrics = r.json()
        except Exception as exc:
            log.warning("slo_check_fetch_failed", error=str(exc))
            return

        for slo in self._slos:
            val = metrics.get(slo.metric)
            if val is None:
                continue
            breaching = self._is_breaching(val, slo)
            was       = self._breach_state.get(slo.name, False)

            if breaching and not was:
                self._fire_alert(slo, val, "opened")
            elif not breaching and was:
                self._fire_alert(slo, val, "resolved")

            self._breach_state[slo.name] = breaching

    @staticmethod
    def _is_breaching(val: float, slo: SLOTarget) -> bool:
        ops = {
            "lt":  lambda v, t: v >= t,
            "gt":  lambda v, t: v <= t,
            "gte": lambda v, t: v <  t,
            "lte": lambda v, t: v >  t,
        }
        return ops[slo.operator](val, slo.threshold)

    def _fire_alert(self, slo: SLOTarget, val: float, state: str) -> None:
        emoji  = "🔴" if (slo.severity == "critical" and state == "opened") else ("🟡" if state == "opened" else "🟢")
        msg    = f"{emoji} SLO {state.upper()}: {slo.name} = {val} (threshold {slo.operator} {slo.threshold})"

        log.warning("slo_alert", name=slo.name, val=val, threshold=slo.threshold, state=state, severity=slo.severity)
        print(msg, flush=True)

        if self._webhook_url:
            asyncio.create_task(self._send_webhook(msg))

    async def _send_webhook(self, text: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(self._webhook_url, json={"text": text})
        except Exception as exc:
            log.warning("webhook_failed", error=str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    tracker = SLOTracker(
        dashboard_api_url=os.getenv("DASHBOARD_API", "http://localhost:8002"),
        poll_interval_s=int(os.getenv("SLO_POLL_INTERVAL", "300")),
        webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
    )
    asyncio.run(tracker.run_forever())
