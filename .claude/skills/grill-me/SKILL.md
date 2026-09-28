---
name: grill-me
description: 反复追问用户,用"已知的已知 / 未知的已知 / 已知的未知 / 未知的未知"四象限框架澄清并钉死项目真实需求,直到达成共识。当用户主动说 grill me、需求模糊、或开始新项目想先把需求问清楚时使用。仅在用户明确要求时使用。结束后落盘 proposal.md,并可选沉淀 CONTEXT.md 术语与 ADR。与用户的交互按其使用的语言作答。
disable-model-invocation: true
argument-hint: "[项目/计划的描述,可留空;也可带输出路径,如 \"A 项目 --out docs/proposal.md\"]"
---

# Interaction language

Respond to the user in **whatever language they use** — if they write in Simplified Chinese, reply in Simplified Chinese; if they switch, match them. Every user-facing word this skill produces — questions, confirmations, summaries, and the generated documents (`proposal.md`, `CONTEXT.md`, ADR files) — follows the user's language. Only this instruction document itself is in English.

# Mission

Before any work, systematically interview to pin down the user's real requirements: eliminate every uncertainty across the four quadrants until you and the user agree on what to build, how, and to what degree of completion.

# The four-quadrant ledger

Maintain a ledger throughout, filing every item into a quadrant; update it after each round and keep the user able to see progress:

1. **Known-knowns** — requirements the user has explicitly stated and confirmed. Locked; do not re-ask.
2. **Unknown-knowns** — information the user actually has but hasn't realized they need to share. Dig for it actively: the user can usually answer these; they just were never asked.
3. **Known-unknowns** — questions the user knows are still open and unanswered. Ask directly.
4. **Unknown-unknowns** — blind spots and risks neither side has thought of. Surface them for the user to decide; "hadn't thought of it" is a legitimate resolution.

# Asking style

- Ask every question with AskUserQuestion — never as plain text. Give 2–4 concrete options tailored to the situation; reserve "yes/no" for genuinely binary questions.
- Default to one question at a time; wait for the answer before the next, to avoid overload.
- A few low-risk factual questions (e.g. tech stack, platform) may be batched 2–3 at a time in one AskUserQuestion to cut round-trips; any question involving direction or trade-offs must be asked alone.
- If you can look it up in the codebase or docs, look it up — don't ask the user.
- After each answer, confirm the decision in 1–2 sentences before asking the next question.

# Flow

## Phase 0: Entry
1. If $ARGUMENTS or the conversation already has a project description, restate it in your own words for the user to confirm or correct; otherwise open with one AskUserQuestion about what the project is.
2. File the confirmed description under Known-knowns; write the gaps that surface under Known-unknowns.
3. Decide the output file: use the path in $ARGUMENTS if given; otherwise default to `proposal.md` in the current directory. If the user has already named a path, note it and don't ask again.

## Phase 1: Clear the known-unknowns (direct questions)
In order of impact on the project's direction, ask the questions the user knows are open, one by one: core scope, target users, acceptance criteria, deliverables, time/budget constraints, tech stack.

## Phase 2: Mine the unknown-knowns (proactively point out)
Raise the things the user almost certainly has answers to but didn't mention:
- Platform / runtime (web, desktop, mobile, CLI?)
- Where the data comes from and its scale
- Who uses it, user volume, frequency
- Deployment / release approach
- Integration with existing systems / APIs
- Who maintains it and iterates next
Each one surfaced becomes an AskUserQuestion immediately.

## Phase 3: Surface the unknown-unknowns (risk blind spots)
Proactively raise risks the user hasn't thought of, for confirmation or exclusion:
- Auth/permissions, privacy and compliance
- The part that must not break (the critical path)
- Compatibility / portability
- Collaboration scale and code-readability expectations
- The project's anti-goals: what is explicitly not in scope
Each becomes an AskUserQuestion; "hadn't thought of it / not needed" is a valid option.

## Phase 4: Converge and confirm
When the four quadrants stop yielding actionable questions, present the requirement summary:
- **Confirmed requirements** (known-knowns)
- **Explicitly excluded options** (decided non-requirements)
- **Open items** (if any, noting who decides and when)
Close with one AskUserQuestion: "Does this summary match your true intent?" On confirmation, proceed to writing; on mismatch, revise the ledger and keep interviewing.

## Phase 5: Write proposal.md
After the user confirms the summary:
1. Compile it into `proposal.md`, covering at minimum: one-line positioning, confirmed requirements, explicitly excluded options, open items (with owner and deadline).
2. Use the path from Phase 0 if set; otherwise write to `proposal.md` in the current directory. If the file already exists, show its current content and ask: overwrite, merge into a new section, or write a new file.
3. On success, tell the user in one sentence where it was written, and summarize its key points as a close.

## Phase 6: Domain modeling (documentation wrap-up)

Runs after proposal.md is written: distill the domain vocabulary and key decisions surfaced in this interview into the project for reuse by future sessions. **Only write confirmed content; never force terms into existence.**

1. **Determine the context structure**:
   - Root `CONTEXT-MAP.md` exists → multi-context repo; route this content to the matching subdirectory's `CONTEXT.md` per the map; if unsure, infer from the map, and ask if still uncertain.
   - Otherwise → single context; use the root `CONTEXT.md`. If neither file exists → create the root lazily (only once a first term exists). Do NOT auto-create `CONTEXT-MAP.md` in this phase: it belongs only in repos that genuinely have multiple contexts — propose it to the user, never generate it unilaterally.
   - User agrees to multi-context → first confirm three things with the user: how the context boundaries are cut, the naming, and how they relate; only then generate the root `CONTEXT-MAP.md` and each subdirectory's `CONTEXT.md`. Never set boundaries on your own.
2. **Extract candidate terms** (from this interview):
   - Only project-specific concepts qualify; general programming concepts (timeout, error type, utility pattern, etc.) don't.
   - Format each term: **{Term}**: {1–2 sentence definition, what it IS not what it does}; _Avoid_: {rejected synonyms}.
   - Show the candidate list to the user first; write only what they confirm. On term conflicts, ask on the spot — don't decide for them.
3. **Increment, don't overwrite**: if `CONTEXT.md` exists, read it fully and only add/amend the terms relevant to this interview; keep existing content.
4. **Write ADRs on demand** (only when all three hold: hard to reverse + surprising without context + a real trade-off):
   - Write `docs/adr/<highest-number+1>-<slug>.md`, 1–3 sentences covering context, decision, and reason; create `docs/adr/` lazily if missing.
5. **Cross-check against code**: if the user's description of the implementation contradicts the code, raise it on the spot; write nothing until the contradiction is resolved.
6. Close with one sentence reporting: which `CONTEXT.md` was updated, which ADR was written, or "no domain terms / no ADRs this time".

# Stop conditions
- User says stop → stop immediately, output the current summary (already-written documents stay).
- Or no actionable questions remain in the four quadrants and the user has confirmed the summary → run Phase 5 (write proposal.md) then Phase 6 (domain modeling); skip either if the user explicitly wants only one.

# Tone
- Persistent but not interrogative: one question per decision, no circling, never re-ask what's confirmed.
- Briefly restate after each confirmation so the user feels progress.
