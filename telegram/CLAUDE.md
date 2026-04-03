# Telegram Interface Context

You are running inside the **YQG_worker1 Telegram bot** on worker1 (Linux Debian 11, amd64). Messages you send will be delivered to the user's Telegram client.

## Response formatting
- Responses are split at **4096 characters** — prefer concise replies
- Use numbered or bulleted lists for multi-step content so splits land cleanly
- **Bold** (`**text**`) and `inline code` render correctly in Telegram
- Tables, horizontal rules (`---`), and most HTML tags do **not** render — avoid them
- LaTeX does not render — write math in plain text (e.g. `omega_mech`, `sqrt(n_th)`)

## Context
Full lab context (experiment, EPICS channels, scripts, hardware) is in the parent `CLAUDE.md` one directory up. You have access to the labutils codebase and the Y1:DMD control system on this machine.

## Sessions and memory
This Telegram interface maintains named conversation sessions (e.g. `/session lab`, `/session coding`). Each session has its own Claude Code session ID and conversation history. Memory files for Telegram sessions are stored separately from the main labutils CLI memory — use this to track things specific to how you interact with the user over Telegram.
