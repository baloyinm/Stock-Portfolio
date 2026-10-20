# CLAUDE.md: stock-portfolio

<!-- reuse-rule:start v1.1.0 — managed by baloyinm/Re-usable-Component/rules/reuse-rule.md; update it there, not here -->
## Re-usable components: check, reuse or modify, then build (Jarvis rule)

The shared library is the private repo **baloyinm/Re-usable-Component** (`CATALOG.md` lists everything; `NAMING.md` sets names). Goal: more planning, less work. Nothing is built from scratch if a library component already does the job, or can be modified to do it.

**Reuse check: required before building anything (every task, every session)**
1. Plan first: list the parts the task needs.
2. Search the library for every part: clone or pull `baloyinm/Re-usable-Component`, run `python3 tools/find.py --plan "<part one>" "<part two>" ...` (it shows the closest components for each part, strong and partial matches), and read `CATALOG.md`. Open the README of every strong or partial match.
3. Decide each part, in this order:
   - **USE**: a component already does the job. Use it as it is.
   - **MODIFY**: a component does part of the job, or would with a new option or setting. Change that component in the library (keep the projects already using it working, raise its version), then use it.
   - **BUILD**: only when nothing comes close, or when changing the closest one would break the projects using it or take more work than building new.
4. Write the reuse check into your plan before writing any code, one line per part: `part · closest component (or none) · USE / MODIFY / BUILD · why`. No building starts until it is written. For every BUILD, say why MODIFY was not possible.
5. Record what you used in the commit: `uses <name> v<version>`.

**While building**
6. Build every BUILD part, and anything else that could be used again (a screen part, chart, template, script, connector, sync, scheduled-task prompt, business rule), as a self-contained, generic component: settings instead of hard-coded names, amounts or IDs.
7. Name it by `NAMING.md`: `<area>/<subject>-<what-it-is>`, for example `ui/hud-donut-chart` or `integrations/markets-price-feed`.

**When it works: commit to the library**
8. MODIFY parts: commit the change to the library as `improve <name> (from <this project>): <what changed>`, raise its version, and update its README and its `CATALOG.md` line. If old use had to break, raise the major version and say in the README what to change. Never make a parallel copy instead.
9. BUILD parts: run the duplicate check again (`python3 tools/find.py <2 to 4 words describing it>`). If anything now does the same job, even partly, treat it as MODIFY. Otherwise add it in its area folder with a `README.md` (header block from `NAMING.md`) and a small `example/`, add one line to `CATALOG.md`, run `python3 tools/find.py --check`, then commit `add <name> (from <this project>)` and push.
10. Two components that do the same job are never allowed. If you find a pair, merge them and mark the weaker one `deprecated`, pointing to the other.
11. Never put passwords, keys, tokens, client data or personal details in the library.

**At the end of each session**
12. In your final summary, show the reuse check lines and list the components used, modified and added.

**New projects** get this section in their CLAUDE.md on day one, with the shared Jarvis token and model rules.
<!-- reuse-rule:end -->
