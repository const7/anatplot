# Contributing

## Development and checks

Install the package in editable mode with its development and documentation tools
from the repository root:

```bash
uv sync
```

Use these commands during development:

| Task | Command |
| --- | --- |
| Format code | `uv run ruff format .` |
| Check style | `uv run ruff check .` |
| Check formatting | `uv run ruff format --check .` |
| Check types | `uvx mypy anatplot` |
| Run tests | `uv run pytest` |
| Preview docs | `uv run zensical serve` |
| Build docs | `uv run zensical build --clean --strict` |
| Render plot overview | `uv run python docs/scripts/render_plot_overview.py` |
| Build distributions | `uv build` |

The homepage example and tutorials execute during the documentation build; their
figures do not need to be generated separately. The overview is a checked-in
image so it also displays in the GitHub README.

Before requesting review, run the code checks for source changes. For docs or API
changes, build and review the affected pages. For packaging changes, install and
import the built wheel outside the checkout.

Source functions and methods, including private helpers, use type annotations and
NumPy-style docstrings. Describe relevant shapes, units, ordering, ownership, and
side effects. Document constructor parameters on `__init__`.

Add or update tests for changed behavior. Regression tests should reproduce the
original bug. For rendering changes, run graphics tests and inspect fresh figures.
Linux CI uses Xvfb and `PYVISTA_MULTI_SAMPLES=0` so Mesa software rendering
can display translucent surfaces with depth peeling.
Without a graphics context, run `uv run pytest -m "not render"` and note the
missing coverage in the PR.

Use synthetic or attributed public test data. Keep private data and routine
generated outputs out of Git.

## Changing scientific behavior

Keep input requirements in public docstrings and the
[input guide](../use-your-own-data.md), and output behavior in [Save and display](../save-and-display.md).
Rendering must not silently register inputs, infer anatomical membership, or
change the data represented by colors. Input meshes and contact arrays are
copied before plotting; `ROIPlot` retains its input image while regions are
added, so callers must not modify it during that time. Preserve these ownership
contracts when changing data loading or scene construction.

## Issues and pull requests

Search existing issues and PRs. Discuss non-trivial changes in an existing or new
issue before implementation, especially features, public API or dependency
changes, and substantial refactors. Bug reports should include a minimal
reproducer, expected and observed behavior, and relevant versions. Small,
unambiguous fixes may go directly to a PR.

For a PR, work on a branch from `main` and focus on one outcome. Use a draft PR
while the approach or implementation is incomplete. Before requesting review:

- Explain the problem, resulting behavior, and any compatibility or migration
  impact. Link the issue with `Closes #123` only if fully resolved; otherwise use
  `Refs #123`.
- State the validation commands and results, including skipped checks and their
  reasons. Include representative images for rendering changes.
- Update affected documentation for user-visible changes.
  Review the complete diff for unrelated edits and generated files.

Merge after review and applicable checks pass. Squash review fixups; retain
separate commits when they represent useful, independently reviewable changes.

## Commits

Use `type: summary`, without a scope, for commits and PR titles. Keep the summary
concise and specific, for example: `fix: preserve camera state during export`.

Default to a single-line title. Add a body only for non-obvious motivation,
tradeoffs, or compatibility impacts. Do not add a routine `Validation:` section.

- `feat`: new functionality.
- `fix`: bug fixes.
- `docs`: documentation changes.
- `refactor`: restructuring without behavior changes.
- `chore`: maintenance, including CI, builds, dependencies, and performance tuning.

Choose the type by the main outcome. Include tests with the corresponding fix,
feature, or refactor rather than treating them as a separate commit type. Keep
related documentation together with the change; split independent changes rather
than files. Mark incompatible public changes with `!` before the colon and a
`BREAKING CHANGE:` footer describing the impact and migration.
