# Coding agent instructions

Follow the [contributing guide](docs/development/contributing.md) for workflow,
tooling, and code conventions. Keep domain contracts in public docstrings and
the [input guide](docs/use-your-own-data.md); see [output behavior](docs/save-and-display.md) for
figures and interactive scenes.

## Implementation

Keep changes within scope. Preserve unrelated work and retain hand-written
comments when moving or refactoring code unless they have become inaccurate.

Prefer the smallest complete implementation, cohesive functions, and maintained
libraries. Add abstractions or dependencies only for established needs.

## Code contracts

Validate inputs at public boundaries, then trust internal data. Avoid silent
repairs, broad exception handling, and speculative compatibility layers.

Keep API descriptions in source docstrings.

## Verification and commits

During development, run targeted checks for the changed behavior. Before
committing, complete the applicable checks in the contributing guide once;
repeat them only when later edits invalidate the results.

On macOS, run VTK checks and tutorial-executing docs builds outside the agent's
sandbox, even for offscreen rendering; sandboxed AppKit can abort.

Commit each completed, verified logical change promptly using the contributing
guide's commit format. Include only changes belonging to the task, and leave
completed work uncommitted only when the user requests it.

Report the commit, what changed, the checks performed, and any remaining
limitations. Base completion claims on successful commands and fresh outputs.
