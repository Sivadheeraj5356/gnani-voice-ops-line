# gnani-voice-ops-line

A Hinglish voice reporting line for field operations crews, built on Gnani
AI's speech stack for the **Great Indian AI Internship Challenge**
(deadline Nov 10, 2026, 11:59 PM IST).

## The problem

Distributed field crews - data collection, site work, delivery - end every
day with a supervisor chasing updates over chat. Half the updates never get
written down, and the ones that do arrive as forty unstructured voice notes.

This gives every worker a number to call. They speak in whatever mix of
Hindi and English comes naturally, the call is transcribed, structured into
a site report, and the supervisor reads one dashboard instead.

## Pipeline

```
phone audio
  -> Prisma v2.5 STT (REST, https://api.vachana.ai/stt/v3)
  -> deterministic Hinglish structurer (site, workers, hours, issues)
  -> JSONL store + daily aggregation
  -> supervisor dashboard (HTML) + Timbre v2.5 TTS readback
     (REST, https://api.vachana.ai/api/v1/tts/inference)
```

Gnani models/APIs used (named per the challenge rules):

- **Prisma v2.5** speech-to-text - chosen because it is trained on real
  telephonic audio and code-mixed speech, which is exactly what crew calls
  sound like
- **Timbre v2.5** text-to-speech - reads the daily summary back to callers
  in the same language mix
- **Evon v3.3** (open weights) is the planned upgrade for report
  structuring; the shipped structurer is deterministic on purpose, so the
  demo can never invent ops numbers. Fields the caller didn't mention stay
  empty - a report that hallucinates worker counts is worse than no report.

## Run it

No dependencies, Python 3.10+. Without an API key it runs on a mock provider
so every code path is testable offline:

```bash
python voice_ops_line.py demo          # end-to-end on mock calls
python voice_ops_line.py test          # unittest suite
python voice_ops_line.py serve --port 8080
#   POST /webhook/recording   (audio bytes, or {"transcript": ...} in dev)
#   GET  /dashboard           supervisor view
#   GET  /reports             JSON
```

To go live: register for the challenge, create a key in the Gnani API
dashboard (app.gnani.ai/voice), then:

```bash
export GNANI_API_KEY=...
python voice_ops_line.py serve --port 8080
```

Point a telephony provider's recording webhook at `/webhook/recording`.

## Award categories this targets

- **Mixed Vibes** - the whole interface is code-mixed Hinglish speech
- **Real Deal** - solves a real daily problem for field supervisors
- **Noise Canceller** - crew calls come in on messy telephony audio, which
  is what Prisma is built for

## Submission checklist

- [x] App built against the documented Gnani APIs (Prisma + Timbre named
  above); runs offline on a mock provider - live API exercise pending
  registration credits
- [ ] 60-second demo video: one Hinglish call in, dashboard updates live
- [ ] Public post on LinkedIn or X tagging Gnani AI with
  #GreatIndianAIInternshipChallenge and #GnaniAI
- [ ] Submit before 11:59 PM IST, Nov 10 2026
