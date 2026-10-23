# Changelog fragments

Every PR that lands on `main` adds its CHANGELOG entry here, as one small file,
instead of editing `CHANGELOG.md`. At release, `towncrier build` writes the
fragments into a new `CHANGELOG.md` section and deletes them. Two PRs never
touch the same file, so the changelog cannot conflict between open PRs.

## Name

`<PR number>.<type>.md`, for example `321.fixed.md`. Open the PR first, then
add the fragment under its number in the next push. A PR with two separate
entries of one type adds `321.fixed.md` and `321.fixed.1.md`.

| Type          | Heading       | For                                                       |
|---------------|---------------|-----------------------------------------------------------|
| `changed`     | Changed       | a change to existing behaviour                            |
| `added`       | Added         | a new indicator, parameter or function                    |
| `deprecated`  | Deprecated    | a deprecation that ships in this release                  |
| `removed`     | Removed       | a removal                                                 |
| `fixed`       | Fixed         | a bug fix                                                 |
| `performance` | Performance   | a speed-up with unchanged results                         |
| `docs`        | Documentation | documentation only                                        |
| `dev`         | Development   | tests, CI, tooling and contributor process                |
| `upgrading`   | Upgrading     | the maintainer's upgrade summary, written at release, named `+<topic>.upgrading.md` |

## Content

Markdown, written for library users: what changed and what a user must do.
One to three sentences, at most 1,000 characters. The details stay in the PR.
Do not indent a second line; towncrier indents it under the bullet.
towncrier adds the bullet and the PR number, so write neither.

```markdown
**`kc(scalar=...)` keeps the default's column name**: `kc(scalar=2)` named its columns `KCLe_20_2.0`; it now names them `KCLe_20_2`, as `kc()` does.
```

An entry that changes results or starts raising for existing calls starts with
`**BREAKING` (AGENTS.md rule 13) and says what the user must change:

```markdown
**BREAKING — `sma(length=0)` raises `ValueError`** instead of returning `SMA_10`. Pass a length > 0, or `None` for the default.
```

If a later PR makes an unreleased entry untrue, that PR edits the entry's
fragment. `tests/test_changelog.py` checks the names, the length and the
**BREAKING** form. A PR that no reader of the changelog needs to know about (a
typo, a version pin, a test refactor) gets the `skip-changelog` label, and the
`Changelog` check then skips; a change contributors must know about (a new CI
job, a new rule) is a `dev` fragment.
