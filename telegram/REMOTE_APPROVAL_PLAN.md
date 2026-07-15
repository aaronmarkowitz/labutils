# Remote permission approval via Telegram — design plan

## Problem

Claude Code runs under three different accounts across two machines (worker1,
lab workstation; Theory, laptop): a Claude Pro subscription, AWS Bedrock
(bills most usage), and a separate Claude Console (API key) account. The goal
is to approve/deny Claude Code's permission prompts remotely (e.g. from a
phone) when away from the machine running the session.

Anthropic's native remote-approval mechanisms don't cover all three:

| Mechanism | Pro (claude.ai OAuth) | Console (API key) | Bedrock |
|---|---|---|---|
| Remote Control + mobile push | Yes | No (API keys unsupported) | No (Bedrock excluded) |
| Channels (Telegram/Discord/iMessage relay) | Yes | Yes | No (Bedrock excluded) |

Bedrock — the majority of actual usage — is excluded from every native path.

**Hooks are the one mechanism that isn't backend-gated.** `PreToolUse` /
`PermissionRequest` hooks are a local CLI feature; nothing in the hooks spec
references which model provider or auth mode is in use. This plan uses hooks
to build a Telegram-based approval relay that works identically across all
three accounts.

## Chosen architecture

- **Hook event: `PermissionRequest`**, not `PreToolUse`. `PermissionRequest`
  fires specifically "when a permission dialog appears" — i.e. exactly when a
  session would otherwise stall waiting on you — rather than on every tool
  call regardless of whether approval was needed.
- **Decision channel:** hook returns
  `{"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": "allow"|"deny"}}}`
  on stdout (exit 0), or `exit 2` to deny. Hook execution is synchronous —
  Claude Code blocks on the hook process, so the script itself can block while
  it waits for a human to respond (default hook timeout 600s; make it
  configurable, target ~10–30 min).
- **Fail-closed:** if nobody answers before the timeout, deny. Silence should
  never resolve to allow. The user can retry the action once back at the
  machine.
- **Two independent Telegram bots, one per machine** (`worker1-approvals-bot`,
  `theory-approvals-bot`), each with its own BotFather token, each paired only
  to the user's Telegram account. Rationale: Telegram allows exactly one
  `getUpdates` long-poll consumer per bot token. A single shared bot would
  require the two machines to coordinate polling (a central broker), which
  was considered and rejected — the machine generating the permission request
  is, by definition, powered on and reachable to the internet at that moment
  (it's actively running the session), so there's no need for one machine to
  depend on the other's uptime. Two bots means two fully independent, symmetric
  hook installations with zero cross-machine coupling. Trade-off: approvals
  show up as two separate Telegram chats rather than one merged thread —
  accepted.
- **Scope:** every permission prompt, on every session, on both machines,
  regardless of which of the three accounts is active. No matcher filtering
  by tool/command — the point is that any prompt that would otherwise stall
  the session should be able to reach the user's phone.
- **Hook install location:** user-level settings
  (`~/.claude/settings.json`) on each machine, so it applies to every session
  on that machine rather than being scoped to one project.

## What the hook script needs to do (per machine, identical logic, different token)

1. Read the `PermissionRequest` hook JSON from stdin (session/tool/input
   context — capture enough to make the Telegram message legible: tool name,
   command/file/input being requested, which machine, which session).
2. `sendMessage` to that machine's bot with an inline Approve/Deny keyboard,
   embedding a unique request ID in the callback data.
3. Long-poll `getUpdates` (this process is the sole consumer of its bot's
   token, so no conflict) filtered to callback queries matching that request
   ID, with an overall deadline.
4. On Approve: answer the callback (so the button UI updates), print the
   allow JSON, exit 0.
5. On Deny or timeout: print/return deny, exit 0 (fail-closed) or exit 2.

## Open items to flesh out during implementation

- Bot token storage (env var / secrets file, consistent with how
  `telegram_claude_bot.py` already reads `TELEGRAM_BOT_TOKEN` from an
  EnvironmentFile) — use a **separate** env var name per machine's approval
  bot so it can't collide with the existing chat/digest bot's token.
  These are new bots, unrelated to the existing `telegram_claude_bot.py` /
  `arxiv_digest.py` bot, so no conflict with that already-running service.
- Message content: include hostname, tool name, and a truncated/summarized
  view of the tool input (full `Bash` commands, file paths for
  `Read`/`Write`/`Edit`, etc.) so an approval can be made from the phone
  without needing more context.
- Logging: log every request/decision/timeout locally (mirror the existing
  `~/.local/share/claude-telegram-bot/` logging convention) for audit.
- Testing: dry-run against a low-stakes prompt before relying on it; confirm
  fail-closed behavior actually blocks the tool (test a deliberate timeout).
- Packaging: likely a small standalone Python script (stdlib `urllib` is
  enough — no need for `python-telegram-bot`, since this only needs
  `sendMessage`/`getUpdates`/`answerCallbackQuery`, not a full bot framework)
  registered as the `command` handler for `PermissionRequest` in
  `~/.claude/settings.json`.
- Rollout order: worker1 first (majority of Bedrock usage), then replicate
  for Theory once validated.
