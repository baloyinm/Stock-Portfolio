# CLAUDE.md: stock-portfolio

<!-- reuse-rule:start v1.0.0 — managed by baloyinm/Re-usable-Component/rules/reuse-rule.md; update it there, not here -->
## Re-usable components: search first, build reusable, commit back (Jarvis rule)

The shared library is the private repo **baloyinm/Re-usable-Component** (`CATALOG.md` lists everything; `NAMING.md` sets names). Goal: more planning, less work. Nothing is built from scratch if the library already has it.

**Before building anything (every task, every session)**
1. Plan first: list the parts the task needs.
2. Search the library for each part: clone or pull `baloyinm/Re-usable-Component`, then run `python3 tools/find.py <words>` or read `CATALOG.md`.
3. Use what fits (copy the folder in, keep its README). Say in the commit: `uses <component name> v<version>`.
4. Build from scratch only what the library does not have, and say so in your plan.

**While building**
5. Build anything that could be used again (a screen part, chart, template, script, connector, sync, scheduled-task prompt, business rule) as a self-contained, generic component: settings instead of hard-coded names, amounts or IDs.
6. Name it by `NAMING.md`: `<area>/<subject>-<what-it-is>`, for example `ui/hud-donut-chart` or `integrations/markets-price-feed`.

**When it works: check for duplicates, then commit it to the library**
7. Duplicate check (required): pull the library, run `python3 tools/find.py <2 to 4 words describing it>` and read `CATALOG.md`. If anything does the same job, even partly, do NOT add a new one: extend that component (new option, raise its version) and commit `improve <name> (from <this project>)`. Two components that do the same job are never allowed; if you find a pair, merge them and mark the weaker one `deprecated` pointing to the other.
   Then add it to `baloyinm/Re-usable-Component` in its area folder with a `README.md` (header block from `NAMING.md`) and a small `example/`; add one line to `CATALOG.md`; run `python3 tools/find.py --check`; commit `add <name> (from <this project>)` and push.
8. Improved a library component here? Commit the improvement back: `improve <name> (from <this project>): <what changed>`, and raise its version.
9. Never put passwords, keys, tokens, client data or personal details in the library.

**At the end of each session**
10. In your final summary list: components used, components added, components improved.

**New projects** get this section in their CLAUDE.md on day one, with the shared Jarvis token and model rules.
<!-- reuse-rule:end -->
