# Workflow Rules

- All AI-generated code goes on `claude/*` branches. Never commit directly to main.
- Claude Code is the primary implementation agent. Cursor is for debugging only.
- Always read the relevant `docs/components/0X-*.md` spec before implementing a component.
- Every new Python module must have a corresponding test file before merging.
- Reference component numbers in commits: "Add actor memory decay (C4)".
- Follow implementation order: C1 → C2 → C3 → C4 → C5 → C6. See docs/components/README.md.
- Update CLAUDE.md (root or subdirectory) whenever a new convention is established.
- After every 5th commit, re-read root CLAUDE.md. Check that commands are accurate and
  conventions match actual code. Update or fix any drift.
