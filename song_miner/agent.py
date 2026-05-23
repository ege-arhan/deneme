from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass
class AgentObservation:
    kind: str
    detail: str
    created_at_utc: str


class LocalLLMAgent:
    """Compact observer/guide/tracker agent.

    - Observer: records pipeline/runtime observations
    - Guide: emits practical next-step suggestions
    - Tracker: stores action timeline for local audits
    - Lyrics helper: can attempt web-assisted lyric extraction

    Uses local Ollama-compatible endpoint and Gemma family model by default.
    """

    def __init__(self, out_dir: str = "data", model: str | None = None) -> None:
        self.out_dir = out_dir
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "gemma4:4b")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.observations: list[AgentObservation] = []
        self.events_path = Path(out_dir) / "agent_events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, kind: str, detail: str) -> None:
        obs = AgentObservation(
            kind=kind,
            detail=detail,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.observations.append(obs)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obs.__dict__, ensure_ascii=False) + "\n")

    def summarize_guidance(self, record: dict[str, Any]) -> str | None:
        prompt = {
            "role": "user",
            "content": (
                "You are a compact music-data assistant. Provide 3 short suggestions "
                "to improve dataset quality and model training readiness.\n\n"
                f"Record:\n{json.dumps(record, ensure_ascii=False)[:6000]}"
            ),
        }
        return self._chat([prompt])

    def fetch_lyrics_via_web(self, title: str, artist: str | None = None) -> str | None:
        q = f"{title} {artist or ''} full lyrics"
        self.observe("lyrics_search", f"search={q}")
        html = requests.get(
            "https://duckduckgo.com/html/", params={"q": q}, timeout=20
        ).text
        links = []
        for token in html.split('href="'):
            if "lyrics" in token.lower() and token.startswith("http"):
                links.append(token.split('"', 1)[0])
        for link in links[:5]:
            try:
                page = requests.get(link, timeout=20).text
                snippet = self._chat(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Extract only the song lyrics from the raw html below. "
                                "If not confidently present, return NONE.\n\n"
                                f"URL: {link}\nHTML:\n{page[:15000]}"
                            ),
                        }
                    ]
                )
                if snippet and snippet.strip().upper() != "NONE" and len(snippet.split()) > 40:
                    self.observe("lyrics_extracted", f"url={link}")
                    return snippet.strip()
            except Exception as exc:  # best-effort branch
                self.observe("lyrics_error", f"url={link} error={exc}")
        return None

    def export_observations(self) -> list[dict[str, str]]:
        return [o.__dict__ for o in self.observations]


    def check_health(self) -> dict[str, str | bool]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return {"ok": True, "model": self.model, "base_url": self.base_url}
        except Exception as exc:
            return {"ok": False, "model": self.model, "base_url": self.base_url, "error": str(exc)}



    def plan_json(self, goal: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        prompt = {
            "role": "user",
            "content": (
                "Return strict JSON with keys: goal, steps[]. Each step has id,name,success_criteria,retry_hint. "
                f"Goal: {goal}. Context: {json.dumps(context or {}, ensure_ascii=False)[:2000]}"
            ),
        }
        txt = self._chat([prompt])
        if not txt:
            return {"goal": goal, "steps": []}
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            return json.loads(txt[start:end+1])
        except Exception:
            return {"goal": goal, "steps": []}

    def verify_record(self, record: dict[str, Any]) -> dict[str, Any]:
        issues = []
        if record.get("needs_review"):
            issues.append("low_match_score")
        if not record.get("lyrics"):
            issues.append("missing_lyrics")
        if not record.get("features") and not record.get("duplicate"):
            issues.append("missing_features")
        decision = "accept" if not issues else ("retry" if len(issues) == 1 else "review")
        return {"decision": decision, "issues": issues}

    def _chat(self, messages: list[dict[str, str]]) -> str | None:
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content")
        except Exception:
            return None
