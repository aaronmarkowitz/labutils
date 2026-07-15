# Remote permission approval via Telegram — install & operate

Implementation of `REMOTE_APPROVAL_PLAN.md`. The relay is a single Claude Code
`PermissionRequest` hook that sends every permission prompt to a per-machine
Telegram bot and blocks until you answer from your phone. Fail-closed: silence
or Deny → deny.

**The code lives in the `claude-config` repo (`~/.claude/`), not here.** This
doc is the operator's guide; `labutils/telegram/` is just where the design +
these instructions are tracked.

## Files (in `~/.claude/`, shared across machines via claude-config)

| File | Role |
|---|---|
| `hooks/telegram-approval.py` | The hook. stdlib-only (`urllib`); no deps. |
| `hooks/telegram-approval-toggle.sh` | `on`/`off`/`status` — mute the relay mid-session. |
| `hooks/setup-approval-bot.sh` | One-time interactive bot/token/chat-id setup. |
| `settings.json` → `hooks.PermissionRequest` | Registers the hook (shared, no matcher = every tool). |

## Per-machine config (NOT in git — outside any repo)

`~/.config/claude-approval-bot/secrets.env` (chmod 600):
```
CLAUDE_APPROVAL_BOT_TOKEN=<from BotFather>
CLAUDE_APPROVAL_CHAT_ID=<your Telegram chat id, captured at setup>
```
Same var names on both machines, different values. Written by
`setup-approval-bot.sh`. **This is the ONLY per-machine step** — the hook
script and its `settings.json` registration are shared through the repo.

### Fail-safe: a machine with no `secrets.env` is a no-op

`telegram-approval.py` calls `load_config()` first; if `secrets.env` is missing
it returns exit 1 (non-blocking) *before any network call*, and Claude Code
shows the normal terminal dialog. So the shared `settings.json` registration is
harmless on any machine that hasn't set up a bot yet — no lockout, no
regression, no need to gate the registration per machine.

## Behavior

- **Most tools** → one message with **Approve / Deny** buttons. Tap to decide.
  On resolution the message is edited in place to a permanent
  ✅ APPROVED / ❌ DENIED / ⏱ TIMED OUT record and its buttons removed.
- **`AskUserQuestion`** → the real question(s) and real options are relayed
  (one message per question):
  - single-select: tap an option;
  - multi-select: toggle options (☑/⬜), then **Submit**;
  - custom answer ("Other"): **reply** to the question message with free text.
  The answer is returned via `decision.updatedInput.answers`, so the tool runs
  already-answered and the terminal prompt is skipped — it never becomes a
  remote bottleneck. On timeout it degrades to the normal terminal prompt.
- **Fail-closed** on all non-question tools: silence past the internal deadline
  (570 s, safely inside Claude's 600 s hook kill-timeout) or an explicit Deny →
  deny. A killed-on-timeout process would otherwise be read as "non-blocking
  error → show dialog", so the script always self-denies *before* that kill.
- **Concurrency**: Telegram permits one `getUpdates` consumer per bot token, so
  all in-flight prompts on a machine pool through one flock-guarded poll loop.
  N simultaneous prompts share one poll — no 409, no rate multiplication.

## Operate

```bash
# Mute the relay (instant terminal dialogs, zero Telegram traffic) — e.g. while
# sitting at the machine:
bash ~/.claude/hooks/telegram-approval-toggle.sh off
# Re-enable right before stepping away:
bash ~/.claude/hooks/telegram-approval-toggle.sh on
bash ~/.claude/hooks/telegram-approval-toggle.sh status
```

Logs: `~/.local/share/claude-telegram-bot/approval-hook.log`
(one line per request/decision/timeout).

Tunables (env vars, rarely needed):
`APPROVAL_HOOK_DEADLINE_S` (default 570), `APPROVAL_HOOK_POLL_S` (default 3).

## Install on a new machine (e.g. worker1)

Prereq: the machine already has `~/.claude/` on the `claude-config` repo with
the branch containing `hooks/telegram-approval.py` + the `settings.json`
registration merged (auto-pull brings this on session start once merged to
`master`).

1. **Create the bot** (Telegram app, on your phone/desktop):
   message **@BotFather** → `/newbot` → give it a name (e.g. "worker1
   Approvals") and a globally-unique username ending in `bot` (e.g.
   `worker1_approvals_bot`). BotFather replies with a token.
2. **Write the config** — in a terminal *on that machine*, NOT through Claude
   (keeps the token out of any session transcript):
   ```bash
   bash ~/.claude/hooks/setup-approval-bot.sh
   ```
   Paste the token when prompted (input hidden). Then send `/start` (or any
   message) to your new bot in Telegram and press Enter — the script fetches
   and saves your chat id.
3. **Prove fail-closed** before relying on it, with a shortened deadline and no
   button tap:
   ```bash
   echo '{"session_id":"t","tool_name":"Bash","tool_input":{"command":"echo hi"},"cwd":"/tmp"}' \
     | APPROVAL_HOOK_DEADLINE_S=15 python3 ~/.claude/hooks/telegram-approval.py
   ```
   Expect (after ~15 s, no tap): `{"hookSpecificOutput": ... "behavior": "deny"}`
   and exit 0. Run again and tap Approve → `"behavior": "allow"`.
4. The hook is already registered in the shared `settings.json`, so it's live
   the moment `secrets.env` exists — no further edit needed. Confirm with a real
   permission prompt in a fresh Claude session.

Each machine's bot is fully independent (its own token, its own chat) — zero
cross-machine coupling, per the plan.
