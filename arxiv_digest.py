#!/usr/bin/env python3
"""Daily arXiv digest — fetches new papers, scores with Claude Haiku, sends to Telegram.

Config (read from environment / EnvironmentFile):
  TELEGRAM_BOT_TOKEN          — bot token
  ALLOWED_TELEGRAM_USER_IDS   — comma-separated user IDs; first entry receives the digest

Run manually to test:
  env $(cat ~/.config/claude-telegram-bot/secrets.env | xargs) python3 arxiv_digest.py
"""

import json
import logging
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_USER_IDS = [
    int(x.strip()) for x in os.environ["ALLOWED_TELEGRAM_USER_IDS"].split(",")
]
# For a private chat (DM), Telegram chat_id == the user's numeric ID
DIGEST_CHAT_ID = ALLOWED_USER_IDS[0]

CLAUDE_CMD = "/home/controls/.local/bin/claude"
HAIKU_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
CLAUDE_ENV_EXTRA = {"CLAUDE_CODE_USE_BEDROCK": "1"}

ARXIV_CATEGORIES = [
    "quant-ph",
    "physics.optics",
    "physics.gr-qc",
    "physics.ins-det",
    "astro-ph.IM",
]
MAX_ARXIV_RESULTS = 200
ABSTRACT_MAX_CHARS = 400   # truncate abstracts before sending to Haiku
MAX_PAPERS_TO_SCORE = 150  # cap to avoid excessively long prompts

LOG_DIR = Path.home() / ".local" / "share" / "claude-telegram-bot"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_MESSAGE_LEN = 4096

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "arxiv-digest.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Haiku relevance prompt
# ---------------------------------------------------------------------------
RELEVANCE_PROMPT = """\
You are a research digest assistant for an experimental physicist specializing \
in levitated optomechanics and quantum sensing.

From the list of papers provided, identify the 3 to 5 most relevant papers. \
For each selected paper output it in exactly this format (use --- as a separator between papers, \
no other extra text):

TITLE: <full title>
AUTHORS: <author list>
LINK: <URL>
RELEVANCE: <one sentence explaining why this is relevant>
ABSTRACT: <full abstract text>
---

Relevance criteria (in priority order):
1. Levitated particles: optical trapping, magneto-gravitational traps, diamagnetic levitation, \
nanoparticle/microparticle mechanics, feedback cooling of levitated objects
2. Diamond/NV center spin mechanics or hybrid spin-mechanical quantum systems
3. Gravitationally mediated entanglement, large spatial superpositions, macroscopic quantum states, \
decoherence reduction in mechanical systems
4. Measurement backaction, quantum-limited sensing, optomechanical squeezing, \
quantum non-demolition measurement
5. Optimal control in optomechanics or open quantum systems
6. Quantum force sensing, acceleration sensing, quantum-limited measurement
7. Papers by or closely related to (higher priority if author matches): \
Aspelmeyer, Sudhir, Harris (Jack), Moore (David), Chen (Yanbei), Miao (Haixing), \
Martynov, Gardner (James), Hall (Evan), Mazumdar (Anupam), Geraci (Andrew), \
Novotny (Lukas), D'Urso (Brian), Morley (Gavin), Purdy (Tom), Millen (James), \
Ulbricht (Hendrik), Marquardt (Florian), Li (Tongcang), Barker (Peter), Dutt (Gurudev), \
Lukin (Mikhail), Kippenberg (Tobias), Painter (Oscar), Khalili (Farid), \
Mavalvala (Nergis), Aggarwal (Nancy)

If fewer than 3 papers are clearly relevant, return however many you find and append a line:
TOTAL_REVIEWED: <N>

Papers to review:
"""


# ---------------------------------------------------------------------------
# arXiv fetch
# ---------------------------------------------------------------------------
def fetch_arxiv_papers(lookback_hours: int = 36) -> list[dict]:
    """Fetch recent new submissions from arXiv across target categories."""
    cat_query = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    params = urllib.parse.urlencode({
        "search_query": cat_query,
        "start": 0,
        "max_results": MAX_ARXIV_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    logger.info("Fetching arXiv: %s", url)

    req = urllib.request.Request(url, headers={"User-Agent": "arxiv-digest-bot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml_data = resp.read()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    papers = []
    for entry in root.findall("atom:entry", ns):
        published_str = entry.findtext("atom:published", "", ns)
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if published < cutoff:
            break

        arxiv_id = (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]
        title = (entry.findtext("atom:title", "", ns) or "").replace("\n", " ").strip()
        abstract = (entry.findtext("atom:summary", "", ns) or "").replace("\n", " ").strip()
        authors = ", ".join(
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        )
        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "link": f"https://arxiv.org/abs/{arxiv_id}",
        })

    logger.info("Fetched %d recent arXiv papers (lookback=%dh)", len(papers), lookback_hours)
    return papers


# ---------------------------------------------------------------------------
# Inspire-HEP fallback
# ---------------------------------------------------------------------------
def fetch_inspirehep_papers() -> list[dict]:
    """Fallback: recent published papers from Inspire-HEP covering PRL, PRA, CQG, Nature, etc."""
    query = (
        "levitated optomechanics OR (NV center AND mechanical) OR "
        "gravitational entanglement macroscopic quantum OR "
        "(quantum sensing AND optomechanics) OR (levitated AND decoherence)"
    )
    params = urllib.parse.urlencode({
        "sort": "mostrecent",
        "size": 30,
        "page": 1,
        "q": query,
        "fields": "titles,authors,abstracts,arxiv_eprints,dois",
    })
    url = f"https://inspirehep.net/api/literature?{params}"
    logger.info("Fetching Inspire-HEP fallback: %s", url)

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    papers = []
    for hit in data.get("hits", {}).get("hits", []):
        meta = hit.get("metadata", {})
        title_list = meta.get("titles", [])
        title = title_list[0].get("title", "") if title_list else ""
        abstract_list = meta.get("abstracts", [])
        abstract = abstract_list[0].get("value", "") if abstract_list else ""
        authors = ", ".join(
            a.get("full_name", "") for a in meta.get("authors", [])[:8]
        )
        arxiv_list = meta.get("arxiv_eprints", [])
        arxiv_id = arxiv_list[0].get("value", "") if arxiv_list else ""
        doi_list = meta.get("dois", [])
        if arxiv_id:
            link = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi_list:
            link = f"https://doi.org/{doi_list[0]['value']}"
        else:
            link = ""
        if title and abstract:
            papers.append({"title": title, "authors": authors, "abstract": abstract, "link": link})

    logger.info("Fetched %d Inspire-HEP papers", len(papers))
    return papers


# ---------------------------------------------------------------------------
# Haiku scoring
# ---------------------------------------------------------------------------
def build_paper_text(papers: list[dict]) -> str:
    lines = []
    for i, p in enumerate(papers[:MAX_PAPERS_TO_SCORE], 1):
        abstract = p["abstract"]
        if len(abstract) > ABSTRACT_MAX_CHARS:
            abstract = abstract[:ABSTRACT_MAX_CHARS] + "..."
        lines.append(
            f"[{i}] TITLE: {p['title']}\n"
            f"    AUTHORS: {p['authors']}\n"
            f"    LINK: {p['link']}\n"
            f"    ABSTRACT: {abstract}"
        )
    return "\n\n".join(lines)


def score_papers_with_haiku(papers: list[dict]) -> str:
    """Send papers to Claude Haiku; returns formatted digest text."""
    full_prompt = RELEVANCE_PROMPT + build_paper_text(papers)
    result = subprocess.run(
        [CLAUDE_CMD, "-p", full_prompt, "--model", HAIKU_MODEL, "--output-format", "text"],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **CLAUDE_ENV_EXTRA},
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Haiku call failed (rc={result.returncode}): {result.stderr[:300]}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Telegram send
# ---------------------------------------------------------------------------
def send_telegram(chat_id: int, text: str) -> None:
    for i in range(0, max(len(text), 1), MAX_MESSAGE_LEN):
        chunk = text[i : i + MAX_MESSAGE_LEN]
        payload = json.dumps({"chat_id": chat_id, "text": chunk}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("Starting daily arXiv digest for %s", today)

    # 1. Fetch arXiv papers
    try:
        papers = fetch_arxiv_papers(lookback_hours=36)
    except Exception as exc:
        logger.error("arXiv fetch failed: %s", exc)
        send_telegram(DIGEST_CHAT_ID, f"arXiv digest error: failed to fetch papers.\n{exc}")
        sys.exit(1)

    used_fallback = False
    if len(papers) < 5:
        logger.info("Fewer than 5 arXiv papers found — switching to Inspire-HEP fallback")
        try:
            papers = fetch_inspirehep_papers()
            used_fallback = True
        except Exception as exc:
            logger.warning("Inspire-HEP fallback also failed: %s", exc)

    if not papers:
        send_telegram(DIGEST_CHAT_ID, f"arXiv digest {today}: no papers found in any source.")
        return

    # 2. Score with Haiku
    try:
        digest = score_papers_with_haiku(papers)
    except Exception as exc:
        logger.error("Haiku scoring failed: %s", exc)
        send_telegram(DIGEST_CHAT_ID, f"arXiv digest error: Haiku scoring failed.\n{exc}")
        sys.exit(1)

    # 3. If < 3 relevant papers and haven't done fallback, try Inspire-HEP supplement
    if not used_fallback and digest.count("TITLE:") < 3:
        logger.info("Fewer than 3 relevant papers — appending Inspire-HEP supplement")
        try:
            fallback_papers = fetch_inspirehep_papers()
            if fallback_papers:
                fallback_digest = score_papers_with_haiku(fallback_papers)
                digest = (
                    digest
                    + "\n\n--- Recent journal papers (Inspire-HEP) ---\n\n"
                    + fallback_digest
                )
        except Exception as exc:
            logger.warning("Inspire-HEP supplement failed: %s", exc)

    # 4. Send
    source_tag = " [journal fallback]" if used_fallback else ""
    header = f"arXiv digest — {today}{source_tag}"
    send_telegram(DIGEST_CHAT_ID, f"{header}\n\n{digest}")
    logger.info("Digest sent to chat_id=%d", DIGEST_CHAT_ID)


if __name__ == "__main__":
    main()
