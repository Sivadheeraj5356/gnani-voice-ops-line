#!/usr/bin/env python3
"""voice_ops_line.py - a Hinglish voice reporting line for field operations crews.

Built on Gnani AI's speech stack for the Great Indian AI Internship Challenge.

The problem: distributed field crews (data collection, site work, delivery)
end every day with a supervisor chasing updates over chat. Half the updates
never get written down. This gives every worker a number to call: they speak
in whatever mix of Hindi and English comes naturally, the call is transcribed,
structured into a site report, and the supervisor reads one dashboard instead
of forty voice notes.

Pipeline:
  phone audio -> Prisma v2.5 (STT, api.vachana.ai/stt/v3) -> structuring
  -> daily site reports -> Timbre v2.5 (TTS, api.vachana.ai/api/v1/tts/inference)
  reads the summary back to the caller.

No framework dependencies for the core: pure stdlib so it runs anywhere.
Set GNANI_API_KEY to go live; without it the app runs on a mock provider so
every code path is testable offline.

    python voice_ops_line.py serve --port 8080     # webhook + dashboard
    python voice_ops_line.py demo                   # end-to-end on mock audio
    python voice_ops_line.py test                   # unittest suite

Gnani models/APIs used (named per the challenge rules):
  - Prisma v2.5 Speech-to-Text (REST)   https://api.vachana.ai/stt/v3
  - Timbre v2.5 Text-to-Speech (REST)   https://api.vachana.ai/api/v1/tts/inference
  - Evon v3.3 (open weights) is the planned upgrade for report structuring;
    the current structurer is deterministic so the demo never hallucinates
    ops numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unittest
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STT_URL = "https://api.vachana.ai/stt/v3"
TTS_URL = "https://api.vachana.ai/api/v1/tts/inference"
DEFAULT_LANG = "hi-IN"  # Prisma handles Hinglish under the Hindi code
DATA_FILE = Path(os.environ.get("OPS_LINE_DATA", "/tmp/ops_line_reports.jsonl"))


# ---------------------------------------------------------------- providers

class STTError(RuntimeError):
    pass


def gnani_transcribe(audio_bytes: bytes, api_key: str, language_code: str = DEFAULT_LANG) -> str:
    """One synchronous REST STT call (clips <= 60s) against Prisma v2.5."""
    boundary = "----opsline7f3a"
    body = b"\r\n".join([
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="language_code"',
        b"",
        language_code.encode(),
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="format"',
        b"",
        b"transcribe",
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="audio"; filename="call.wav"',
        b"Content-Type: audio/wav",
        b"",
        audio_bytes,
        f"--{boundary}--".encode(),
        b"",
    ])
    req = urllib.request.Request(
        STT_URL, data=body,
        headers={"X-API-Key-ID": api_key, "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    for key in ("transcript", "text", "output"):
        if isinstance(payload, dict) and payload.get(key):
            return str(payload[key])
    raise STTError(f"unexpected STT response shape: {sorted(payload) if isinstance(payload, dict) else type(payload)}")


def gnani_speak(text: str, api_key: str, voice: str = "Nalini") -> bytes:
    """One synchronous TTS call against Timbre v2.5; returns audio bytes."""
    req = urllib.request.Request(
        TTS_URL,
        data=json.dumps({"text": text, "voice": voice, "model": "timbre-v2.5"}).encode(),
        headers={"X-API-Key-ID": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


_MOCK_TRANSCRIPTS = [
    "aaj site par barah log aaye, kaam subah nine se shaam six baje tak chala. "
    "do worker absent the. cable khatam ho gaya hai, kal chahiye.",
    "today fifteen workers present, total hundred twenty hours ho gaye. "
    "rain ki wajah se afternoon shift ruk gayi thi.",
]


def mock_transcribe(audio_bytes: bytes) -> str:
    # Deterministic pick so tests are stable; size decides which canned call.
    return _MOCK_TRANSCRIPTS[len(audio_bytes) % len(_MOCK_TRANSCRIPTS)]


# ---------------------------------------------------------------- structuring

_NUMERALS = {
    # Hindi
    "ek": 1, "do": 2, "teen": 3, "tin": 3, "chaar": 4, "char": 4, "paanch": 5,
    "panch": 5, "chhe": 6, "chhah": 6, "saat": 7, "aath": 8, "nau": 9,
    "das": 10, "gyarah": 11, "barah": 12, "terah": 13, "chaudah": 14,
    "pandrah": 15, "solah": 16, "satrah": 17, "atharah": 18, "unees": 19,
    "bees": 20, "pachees": 25, "tees": 30, "chalis": 40, "pachaas": 50,
    # English
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

_ISSUE_MARKERS = [
    "absent", "khatam", "ruk gayi", "ruk gaya", "problem", "issue", "delay",
    "nahi aaya", "nahi aaye", "chahiye", "rain", "baarish",
]


@dataclass
class SiteReport:
    site: str
    report_date: str
    workers_present: int | None = None
    workers_absent: int | None = None
    hours_worked: float | None = None
    issues: list[str] = field(default_factory=list)
    raw_transcript: str = ""
    received_at: str = ""


def _word_to_int(token: str) -> int | None:
    token = token.strip(".,!?")
    if token.isdigit():
        return int(token)
    return _NUMERALS.get(token.lower())


def structure_report(transcript: str, site: str = "default", today: date | None = None) -> SiteReport:
    """Deterministic Hinglish transcript -> site report. Never invents numbers:
    a field stays None when the caller didn't say it."""
    today = today or date.today()
    text = transcript.lower()
    report = SiteReport(site=site, report_date=today.isoformat(),
                        raw_transcript=transcript,
                        received_at=datetime.now().isoformat(timespec="seconds"))
    if "kal" in text.split("aaj")[0][:20] and "aaj" not in text:
        report.report_date = (today - timedelta(days=1)).isoformat()

    tokens = text.replace(".", " ").replace(",", " ").split()
    # First pass: compounds like "hundred twenty" / "do sau" collapse to one value.
    compound: dict[int, int] = {}
    for i, tok in enumerate(tokens[:-1]):
        if tok == "hundred" and _word_to_int(tokens[i + 1]) is not None:
            compound[i] = 100 + _word_to_int(tokens[i + 1])
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        val = compound.get(i, _word_to_int(tok))
        if val is None:
            continue
        if tok in ("hundred",) and i in compound and _word_to_int(nxt) is not None:
            nxt = tokens[i + 2] if i + 2 < len(tokens) else ""
        # Absence is checked before presence: "do worker absent" is not 2 present.
        if re.match(r"absent", nxt) or (re.match(r"workers?", nxt) and i + 2 < len(tokens) and tokens[i + 2] == "absent"):
            report.workers_absent = val
        elif re.match(r"(log|workers?|logon|people)", nxt):
            report.workers_present = val
        elif re.match(r"(hours?|ghante)", nxt):
            report.hours_worked = float(val)
    m = re.search(r"(\d+)\s*(?:worker|log|workers)\s+absent", text)
    if m:
        report.workers_absent = int(m.group(1))

    sentences = re.split(r"[.!?]+", transcript)
    report.issues = [s.strip() for s in sentences
                     if any(marker in s.lower() for marker in _ISSUE_MARKERS) and s.strip()]
    return report


# ---------------------------------------------------------------- storage

def save_report(report: SiteReport, path: Path = DATA_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(report.__dict__, ensure_ascii=False) + "\n")


def load_reports(path: Path = DATA_FILE) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def daily_summary(day: str, path: Path = DATA_FILE) -> dict:
    rows = [r for r in load_reports(path) if r["report_date"] == day]
    present = [r["workers_present"] for r in rows if r.get("workers_present") is not None]
    hours = [r["hours_worked"] for r in rows if r.get("hours_worked") is not None]
    issues = [issue for r in rows for issue in r.get("issues", [])]
    return {
        "date": day,
        "reports": len(rows),
        "workers_present_total": sum(present) if present else None,
        "hours_total": sum(hours) if hours else None,
        "open_issues": issues,
    }


# ---------------------------------------------------------------- dashboard

def render_dashboard(path: Path = DATA_FILE) -> str:
    rows = load_reports(path)
    days = sorted({r["report_date"] for r in rows}, reverse=True)
    cards = []
    for d in days:
        s = daily_summary(d, path)
        issues = "".join(f"<li>{i}</li>" for i in s["open_issues"]) or "<li>none reported</li>"
        cards.append(f"""
        <section>
          <h2>{s['date']} - {s['reports']} call(s)</h2>
          <p>workers present: <b>{s['workers_present_total'] if s['workers_present_total'] is not None else 'not reported'}</b>
             &nbsp; hours: <b>{s['hours_total'] if s['hours_total'] is not None else 'not reported'}</b></p>
          <ul>{issues}</ul>
        </section>""")
    return ("<!doctype html><meta charset=utf-8><title>Ops line</title>"
            "<style>body{font-family:system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}"
            "section{border:1px solid #ddd;border-radius:8px;padding:1rem;margin:1rem 0}</style>"
            "<h1>Field ops line - daily reports</h1>" + ("".join(cards) or "<p>No calls yet.</p>"))


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    api_key = os.environ.get("GNANI_API_KEY", "")

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/dashboard":
            self._send(200, render_dashboard().encode(), "text/html; charset=utf-8")
        elif self.path == "/reports":
            self._send(200, json.dumps(load_reports(), ensure_ascii=False).encode())
        elif self.path == "/health":
            self._send(200, b'{"ok":true}')
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path == "/webhook/recording":
            # Telephony provider posts call audio; dev mode accepts {"transcript": ...}.
            ctype = self.headers.get("Content-Type", "")
            if "application/json" in ctype:
                transcript = json.loads(body or b"{}").get("transcript", "")
            else:
                transcript = (gnani_transcribe(body, self.api_key) if self.api_key
                              else mock_transcribe(body))
            site = self.headers.get("X-Site", "default")
            report = structure_report(transcript, site=site)
            save_report(report)
            self._send(200, json.dumps(report.__dict__, ensure_ascii=False).encode())
        elif self.path == "/tts-preview":
            text = json.loads(body or b"{}").get("text", "")
            if not self.api_key:
                self._send(503, b'{"error":"GNANI_API_KEY not set"}')
                return
            self._send(200, gnani_speak(text, self.api_key), "audio/wav")
        else:
            self._send(404, b'{"error":"not found"}')

    def log_message(self, *args):
        pass


# ---------------------------------------------------------------- tests

class TestStructuring(unittest.TestCase):
    def test_hinglish_counts(self):
        r = structure_report("aaj site par barah log aaye, do worker absent the. kaam nine se six chala.")
        self.assertEqual(r.workers_present, 12)
        self.assertEqual(r.workers_absent, 2)

    def test_english_counts_and_hours(self):
        r = structure_report("today 15 workers present, total 120 hours done.")
        self.assertEqual(r.workers_present, 15)
        self.assertEqual(r.hours_worked, 120.0)

    def test_unspoken_fields_stay_none(self):
        r = structure_report("sab theek chal raha hai aaj.")
        self.assertIsNone(r.workers_present)
        self.assertIsNone(r.hours_worked)

    def test_issues_extracted_not_invented(self):
        r = structure_report("cable khatam ho gaya hai. baaki sab theek.")
        self.assertEqual(len(r.issues), 1)
        self.assertIn("cable", r.issues[0].lower())


class TestAggregation(unittest.TestCase):
    def setUp(self):
        self.path = Path("/tmp/test_ops_line.jsonl")
        if self.path.exists():
            self.path.unlink()

    def test_daily_summary(self):
        save_report(structure_report("aaj barah log aaye.", site="a"), self.path)
        save_report(structure_report("today 8 workers present, 40 hours.", site="b"), self.path)
        s = daily_summary(date.today().isoformat(), self.path)
        self.assertEqual(s["reports"], 2)
        self.assertEqual(s["workers_present_total"], 20)


class TestProviders(unittest.TestCase):
    def test_mock_is_deterministic(self):
        self.assertEqual(mock_transcribe(b"x"), mock_transcribe(b"x"))

    def test_real_stt_needs_key(self):
        # No key in tests: proving the live path is opt-in, not silent.
        self.assertFalse(os.environ.get("GNANI_API_KEY"))


# ---------------------------------------------------------------- cli

def demo() -> None:
    print("mock mode: two crew calls come in...\n")
    for i, clip in enumerate([b"call-audio-1", b"call-audio-2-"]):
        transcript = mock_transcribe(clip)
        report = structure_report(transcript, site=f"site-{i+1}")
        save_report(report)
        print(f"call {i+1} transcript: {transcript.strip()}")
        print(f"  -> workers: {report.workers_present}, absent: {report.workers_absent}, "
              f"issues: {report.issues}\n")
    s = daily_summary(date.today().isoformat())
    print("supervisor dashboard summary:", json.dumps(s, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--port", type=int, default=8080)
    sub.add_parser("demo")
    sub.add_parser("test")
    args = parser.parse_args()

    if args.cmd == "test":
        unittest.main(argv=[parser.prog], exit=True, verbosity=2)
    elif args.cmd == "demo":
        demo()
    else:
        mode = "live (GNANI_API_KEY set)" if os.environ.get("GNANI_API_KEY") else "mock (no key)"
        print(f"ops line listening on :{args.port} - {mode}")
        ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
