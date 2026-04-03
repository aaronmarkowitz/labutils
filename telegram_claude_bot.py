#!/usr/bin/env python3
"""YQG_worker1 Telegram bot — bridges Telegram to Claude Code (claude-bedrock).

Config (read from environment / EnvironmentFile):
  TELEGRAM_BOT_TOKEN          — bot token from BotFather
  ALLOWED_TELEGRAM_USER_IDS   — comma-separated numeric Telegram user IDs

State persisted in ~/.claude/telegram_sessions.json:
  {
    "<chat_id>": {
      "active": "<session_name>",
      "sessions": {"<name>": "<claude_session_id>", ...},
      "model": "<friendly_model_name>"
    }
  }
Old schema (chat_id -> bare session_id string) is auto-migrated on first load.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_USER_IDS: frozenset[int] = frozenset(
    int(x.strip()) for x in os.environ["ALLOWED_TELEGRAM_USER_IDS"].split(",")
)

SESSIONS_FILE = Path.home() / ".claude" / "telegram_sessions.json"
LOG_DIR = Path.home() / ".local" / "share" / "claude-telegram-bot"
WORKDIR = Path("/home/controls/labutils/telegram")

CLAUDE_CMD = "/home/controls/.local/bin/claude"
CLAUDE_ENV_EXTRA = {"CLAUDE_CODE_USE_BEDROCK": "1"}
ALLOWED_TOOLS = "Bash,Read,Edit,Write,Glob,Grep,Agent"
TIMEOUT_SECONDS = 300  # 5 minutes
MAX_MESSAGE_LEN = 4096

BEDROCK_MODELS: dict[str, str] = {
    "haiku":  "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "opus":   "us.anthropic.claude-opus-4-6-v1",
}
DEFAULT_MODEL = "sonnet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def _migrate(raw: dict) -> dict:
    """Migrate old schema {chat_id: session_id_str} to new schema."""
    result = {}
    for chat_id, val in raw.items():
        if isinstance(val, str):
            result[chat_id] = {
                "active": "default",
                "sessions": {"default": val},
                "model": DEFAULT_MODEL,
            }
        else:
            result[chat_id] = val
    return result


def load_state() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return _migrate(json.loads(SESSIONS_FILE.read_text()))
        except Exception:
            return {}
    return {}


def save_state(state: dict) -> None:
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSIONS_FILE.write_text(json.dumps(state, indent=2))


def get_chat_state(state: dict, chat_id: str) -> dict:
    """Return (initializing if absent) the per-chat state dict."""
    if chat_id not in state:
        state[chat_id] = {"active": "default", "sessions": {}, "model": DEFAULT_MODEL}
    cs = state[chat_id]
    cs.setdefault("active", "default")
    cs.setdefault("sessions", {})
    cs.setdefault("model", DEFAULT_MODEL)
    return cs


# ---------------------------------------------------------------------------
# Claude runner
# ---------------------------------------------------------------------------
async def _invoke_claude(
    prompt: str, session_id: str | None, model: str
) -> tuple[int, bytes, bytes]:
    """Low-level claude invocation. Returns (returncode, stdout, stderr)."""
    full_model_id = BEDROCK_MODELS.get(model, BEDROCK_MODELS[DEFAULT_MODEL])
    cmd = [
        CLAUDE_CMD,
        "-p", prompt,
        "--allowedTools", ALLOWED_TOOLS,
        "--output-format", "json",
        "--model", full_model_id,
    ]
    if session_id:
        cmd.extend(["--resume", session_id])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=WORKDIR,
        env={**os.environ, **CLAUDE_ENV_EXTRA},
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
    return proc.returncode, stdout, stderr


async def run_claude(
    prompt: str, session_id: str | None, model: str
) -> tuple[str, str | None]:
    """Run claude non-interactively. Returns (response_text, new_session_id).

    If the stored session ID is not found (e.g. stale from a different project
    context), automatically retries without --resume so the user gets a response
    rather than an error. The caller will store the new session ID going forward.
    """
    logger.info("model=%s session=%s prompt=%r", model, session_id, prompt[:60])

    try:
        rc, stdout, stderr = await _invoke_claude(prompt, session_id, model)
    except asyncio.TimeoutError:
        logger.warning("claude timed out after %ds", TIMEOUT_SECONDS)
        return f"Request timed out after {TIMEOUT_SECONDS // 60} minutes.", None

    # Detect stale session and retry fresh
    if rc != 0 and session_id and b"No conversation found" in stderr:
        logger.warning("Session %s not found in this project context — retrying fresh", session_id)
        try:
            rc, stdout, stderr = await _invoke_claude(prompt, None, model)
        except asyncio.TimeoutError:
            logger.warning("claude timed out on retry after %ds", TIMEOUT_SECONDS)
            return f"Request timed out after {TIMEOUT_SECONDS // 60} minutes.", None

    if rc != 0:
        err = stderr.decode(errors="replace")[:500]
        logger.error("claude rc=%d: %s", rc, err)
        return f"Error (rc={rc}):\n{err}", None

    try:
        data = json.loads(stdout.decode())
        response: str = data.get("result", "(empty response)")
        new_session_id: str | None = data.get("session_id")
        logger.info("new_session_id=%s response_len=%d", new_session_id, len(response))
        return response, new_session_id
    except json.JSONDecodeError as exc:
        raw = stdout.decode(errors="replace")[:500]
        logger.error("JSON parse error: %s\nRaw: %s", exc, raw)
        return f"Failed to parse response: {exc}", None


# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------
def authorized(update: Update) -> bool:
    return update.effective_user.id in ALLOWED_USER_IDS


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    await update.message.reply_text(
        "YQG_worker1 Claude Code bot ready.\n\n"
        "Commands:\n"
        "  /session — show current session and list all\n"
        "  /session <name> — switch to or create a named session\n"
        "  /session delete <name> — delete a session\n"
        "  /model — show current model\n"
        "  /model <name> — switch model (haiku / sonnet / opus)\n"
        "  /reset — clear current session context"
    )


async def cmd_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    args = context.args

    state = load_state()
    cs = get_chat_state(state, chat_id)

    # /session  (no args) — status
    if not args:
        active = cs["active"]
        session_names = list(cs["sessions"].keys())
        if session_names:
            names_str = ", ".join(
                f"*{n}*" if n == active else n for n in session_names
            )
        else:
            names_str = "(none yet)"
        await update.message.reply_text(
            f"Active session: {active}\nAll sessions: {names_str}"
        )
        return

    # /session delete <name>
    if args[0] == "delete":
        if len(args) < 2:
            await update.message.reply_text("Usage: /session delete <name>")
            return
        name = args[1]
        if name == cs["active"]:
            await update.message.reply_text(
                f"Cannot delete the active session '{name}'. Switch to another session first."
            )
            return
        if name not in cs["sessions"]:
            await update.message.reply_text(f"Session '{name}' not found.")
            return
        del cs["sessions"][name]
        save_state(state)
        await update.message.reply_text(f"Session '{name}' deleted.")
        return

    # /session <name> — switch to or create
    name = args[0]
    if name == cs["active"]:
        await update.message.reply_text(f"Already on session '{name}'.")
        return
    existed = name in cs["sessions"]
    cs["active"] = name
    save_state(state)
    if existed:
        await update.message.reply_text(f"Switched to session '{name}'.")
    else:
        await update.message.reply_text(f"Created and switched to new session '{name}'.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    chat_id = str(update.effective_chat.id)
    args = context.args

    state = load_state()
    cs = get_chat_state(state, chat_id)

    if not args:
        model = cs["model"]
        await update.message.reply_text(
            f"Current model: {model}\n{BEDROCK_MODELS.get(model, '(unknown ID)')}"
        )
        return

    name = args[0].lower()
    if name not in BEDROCK_MODELS:
        await update.message.reply_text(
            f"Unknown model '{name}'. Available: {', '.join(BEDROCK_MODELS)}"
        )
        return

    cs["model"] = name
    save_state(state)
    await update.message.reply_text(f"Model switched to {name}.")


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return
    chat_id = str(update.effective_chat.id)

    state = load_state()
    cs = get_chat_state(state, chat_id)
    active = cs["active"]

    if active in cs["sessions"]:
        del cs["sessions"][active]
        save_state(state)
        await update.message.reply_text(
            f"Session '{active}' cleared. Next message starts a fresh context."
        )
    else:
        await update.message.reply_text(f"Session '{active}' has no context to clear.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not authorized(update):
        return

    chat_id = str(update.effective_chat.id)
    prompt = update.message.text
    logger.info("chat=%s user=%s prompt=%r", chat_id, update.effective_user.id, prompt[:80])

    state = load_state()
    cs = get_chat_state(state, chat_id)
    active = cs["active"]
    session_id = cs["sessions"].get(active)
    model = cs["model"]

    async def keep_typing() -> None:
        while True:
            await update.message.chat.send_action("typing")
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(keep_typing())
    try:
        response, new_session_id = await run_claude(prompt, session_id, model)
    finally:
        typing_task.cancel()

    if new_session_id:
        cs["sessions"][active] = new_session_id
        save_state(state)

    for i in range(0, max(len(response), 1), MAX_MESSAGE_LEN):
        await update.message.reply_text(response[i : i + MAX_MESSAGE_LEN])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Starting YQG_worker1 bot (long-polling)")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("session", cmd_session))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
