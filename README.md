# AI Pair Engineer

A four-stage AI review pipeline that runs **before** a human opens the pull
request: analyse, generate tests, refactor, then adversarially verify the
refactor against the original.

The point is the last stage. Most AI review tools ask a model to grade code it
also helped write. This pipeline makes the reviewer explicitly hostile, gives it
the original and the refactor side by side, and tells it to default to "not
approved".

```
analyzer ──┬──> tester  ──┐
           └──> refactor ──┴──> reviewer ──> verdict
```

---

## Why the stages are separated

Each stage sees only the context it needs, and only that. This is a token
decision and a correctness one:

| Stage | Receives | Never sees |
|---|---|---|
| Analyzer | source + local static analysis | anything |
| Test Engineer | source + error-handling and security findings | the refactor |
| Refactor Engineer | source + maintainability findings | the tests, and all security findings |
| Reviewer | original + refactor + test names + unresolved high findings | — |

Two of those exclusions are the interesting part:

- **The tester never sees the refactor.** Otherwise it writes tests that
  describe the refactored code, which is how a behavioural regression gets
  locked in as an expected result.
- **The refactorer never sees security findings.** A correctness fix bundled
  into a refactor is the one change a reviewer is least able to verify, so
  those defects are reported under `risks` instead of silently repaired.

---

## Two ways to run it

### Static analysis only, no API key

Real AST analysis. Works offline, and the line numbers are exact because they
come from the parse tree rather than from a model guessing.

```console
$ pair-engineer examples/legacy_service.py --static-only
7 functions
0 classes
  get_user (line 30): complexity 6, undocumented
  build_export_path (line 54): complexity 2, undocumented
  ...
  bare except at line 41
  bare except at line 49
  mutable default at line 92
```

It computes McCabe cyclomatic complexity, maximum nesting depth, argument
counts, return counts, and docstring coverage per function, and flags bare
`except:`, mutable default arguments (including ones that alias a module-level
mutable such as `def f(cache=CACHE)`), unused imports, over-long lines, and
TODO comments.

### The full pipeline

Needs an OpenRouter key.

```console
$ export OPENROUTER_API_KEY=sk-or-...
$ pair-engineer examples/sample.py
```

```
Two correctness defects and one maintainability problem.

HIGH     error_handling   Unvalidated dict access raises KeyError
         at line 3, in `process_users`
         user['age'] is read without checking the key exists.
         fix: Use user.get('age') and skip malformed records.

MEDIUM   complexity       Nested conditionals flatten poorly
         at line 3, in `process_users`
         Four levels of nesting for one filter.
         fix: Use guard clauses.

Quality score 74/100
LOW 1  MEDIUM 1  HIGH 1

Verdict NEEDS_REVIEW  score 72/100  regression risk high
Preserve the KeyError or update the callers that relied on it.
  - Original raised KeyError on a missing age key; the refactor silently skips it.
```

> The verdict block above is real output from the test fixtures. Note that the
> reviewer caught a genuine behaviour change in a refactor that looked correct.

---

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/AwonAziz/AI-Pair-Engineer
cd AI-Pair-Engineer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

cp .env.example .env             # Windows: copy .env.example .env
# then add your key to .env
```

Any OpenRouter model that reliably emits JSON will work:

```bash
export MODEL=anthropic/claude-sonnet-4
```

### The web UI

```bash
streamlit run app.py
```

The Streamlit app is a thin wrapper. It collects input, calls the pipeline, and
renders the result. All review logic lives in the package so the CLI and the UI
cannot drift apart.

---

## CLI

```
pair-engineer <file> | --stdin
              [--language python|javascript|typescript|java]
              [--format text|json|markdown]
              [--model <id>]
              [--fail-on critical|high|medium|low|never]
              [--static-only]
              [--no-color]
```

`--format markdown` produces a GitHub-flavoured table, suitable for a PR
comment.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Clean, or no finding at or above `--fail-on` |
| 1 | A finding at or above the threshold exists |
| 2 | Bad usage: missing file, empty input, unknown language |
| 3 | Misconfiguration, usually a missing API key |
| 4 | The provider failed, or a stage could not produce a valid result |

The distinction matters in CI: `3` means fix your environment, `4` means the
upstream provider is down. Treating both as "review failed" trains people to
ignore the output.

`--fail-on` defaults to `high`. `critical` alone lets serious defects through,
and `low` fails on style preferences, which is how a gate gets ignored.

---

## Using it in CI

```yaml
- name: AI review
  env:
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
  run: pair-engineer src/ --format markdown >> "$GITHUB_STEP_SUMMARY"
```

`.github/workflows/self-review.yml` in this repo runs the tool against its own
diff and comments on the pull request.

---

## Does it actually find anything?

Claims about AI code review are cheap; measurements are not. This repository
ships the harness used to produce its own numbers.

A corpus of files with **22 defects deliberately planted** across 7 files,
including one clean file so false positives are measurable. Four strategies are
run against it:

| Tier | What it is |
|---|---|
| `static` | Local AST analysis. No model, no cost, deterministic. |
| `naive` | One unstructured call: "review this code, return JSON". The baseline. |
| `analyzer` | The real analyzer prompt, no local evidence. |
| `analyzer+static` | The real analyzer prompt, with local evidence. |

```bash
python -m benchmarks validate                        # no API key, runs in CI
python -m benchmarks run --tiers static              # free, no API key
python -m benchmarks run --tiers all --breakdown     # needs a key
python -m benchmarks run --tiers analyzer+static --regressions
```

### Measured, so far

The static tier, on the corpus as it stands:

```
| Tier        | Recall | Groundedness | Findings | Clean-file noise |
|-------------|--------|--------------|----------|------------------|
| static only |   9.1% |       100.0% |        4 |            0.0%  |
```

Read that carefully, because it is unflattering and correct: local analysis
catches 2 of 22 planted defects, because the corpus mostly contains problems
that need semantic understanding. It produces zero noise on the clean file, and
every finding it makes is real. That is the right shape for a free tier — a
narrow, precise detector — and it is the honest baseline the model tiers have
to beat.

Model-tier numbers depend on the model and are published from
[the weekly benchmark run](.github/workflows/benchmark.yml) rather than
hardcoded here, because a stale figure in a README is worse than none.

### Why the scoring is careful

This is the part that decides whether a benchmark means anything.

- **Recall** — of the defects planted, how many were reported. This is the
  headline, because a missed injection is worse than a noisy review.
- **Groundedness** — of the findings produced, what share match a real planted
  defect. This is a *noise signal, not precision*, and the report says so in the
  same document as the number. A finding matching no annotated defect may be a
  genuine problem the corpus did not record; counting those as errors would
  measure the annotations rather than the tool.
- **Clean-file noise** — findings on files with no planted defects. Here a
  finding really is very likely a false positive, so this is the one place
  false-positive counting is meaningful.

Two independent match rules, because each covers a different way to be right: a
finding is credited if it names a line inside the defect's span (`located`), or
if it contains every keyword group in the defect's annotation (`semantic`). A
model that miscounts lines still gets credit for finding the bug.

Every metric carries its own caveat in the output, including the two that would
otherwise be easy to hide: the corpus is Python-only and hand-written by the
same person who wrote the analyzer, which is a real conflict of interest; and at
this corpus size the confidence interval on any recall figure spans several
percentage points, so differences under about ten points are noise.

### The reviewer comparison

The project's headline claim is that an adversarial reviewer — told to default
to "not approved", given both versions side by side — catches behaviour changes
a plain comparison misses. Five regression cases plant exactly one behaviour
change each: an inverted guard clause, a truthiness check that drops a zero, a
deleted authorization branch, a type coercion that adds a `ValueError`, removed
validation.

A case counts as caught only when the reviewer **rejected** the refactor *and*
**named** the specific change. Rejecting for vague reasons, or approving while
noticing something, both score as misses.

`--regressions-baseline` runs the same cases against a plain "compare these two
and tell me if it's safe" prompt. That is the comparison that justifies the
whole design, so it gets measured rather than asserted.

### Adding a case

```
benchmarks/cases/my_case/
    source.py      the file to review, with the defects planted in it
    case.toml      the annotations
```

`python -m benchmarks validate` runs in CI and fails on a line number that has
drifted past the end of the file, a defect with no keywords, duplicate ids, or a
regression pair that differs only in comments. Ground truth rots silently
otherwise, and a benchmark against drifted ground truth reports a confident
number that means nothing.

---

## Design decisions worth arguing about

**Output is validated, not trusted.** Every stage is parsed into a Pydantic
model. A response with an enum outside the allowed set, a `confidence` of 7, or
a missing required field is rejected, not coerced. A confidently wrong review is
worse than no review. On rejection the stage is retried once with an explicit
correction prompt; two failures and the run stops with the offending field named.

**Parsing handles what models actually emit.** Bare JSON, JSON in a ```` ```json ````
fence, unterminated fences from a truncated response, and JSON surrounded by
prose. The balanced-object scanner is string-aware, so a field containing
refactored code with unbalanced braces still parses.

**Transport failures and contract failures are different exception types.**
Retries with exponential backoff plus full jitter, capped at 30 seconds. A 4xx
other than 429 is not retried, because it will still be a 401 on the third
attempt. Jitter matters: without it, stages that fail together retry on the same
tick and re-create the burst that caused the rate limit.

**No import-time side effects.** The API key is resolved on first use. Raising
at import time means a misconfigured app cannot start far enough to tell you the
key is missing, and it makes the package unimportable for anyone without one.

**Quality score is deterministic.** `100 - Σ severity weights`, floored at 0,
with the weights in one dict in `schemas.py` so the UI, the CLI, and the tests
cannot disagree. The reviewer's own score is separate and comes from the model.

**Findings carry line numbers.** A finding without a location is much harder to
act on, so `Location` is part of the schema and the analyzer prompt asks for it
explicitly. The static report's numbers are verified facts, and the prompt tells
the model to trust them over its own estimates.

---

## Security

Submitted code is treated as untrusted input.

- **No execution.** The tool analyses code; it never runs it. Generated tests
  are returned as text, not executed.
- **No shell interpolation.** User code never reaches a shell command. There is
  no `subprocess` call anywhere in the package.
- **No key leakage.** The API key is never included in an error message or a log
  line, and there is a test that asserts this.
- **Assistant text is not logged.** Responses embed the user's submitted source.

If you extend this to execute generated tests, it must run in a container or
sandbox with hard CPU, memory, filesystem, network, and wall-clock limits. Do
not execute untrusted code on the host.

---

## Development

```bash
pip install -e ".[dev]"

pytest              # 398 tests, no API key needed, no network
ruff check .
ruff format --check .
mypy                # strict, covers benchmarks/ too
```

| Check | Command |
|---|---|
| Tests | `pytest` |
| Coverage | `pytest --cov=ai_pair_engineer --cov=benchmarks --cov-report=term-missing` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Types | `mypy` |
| Corpus validity | `python -m benchmarks validate` |
| All of it | `pre-commit run --all-files` |

`examples/` and `benchmarks/cases/` are excluded from linting on purpose. Those
files are intentionally defective, and the defects are the demonstration.
Type annotations were added to `AnalyzerAgent.__init__` rather than left to the
exemption, so only the corpus itself is unchecked.

Measured cost is now real rather than assumed: `ask_llm` reports the token
counts the provider returns, each stage accumulates its own usage, and the
pipeline trace totals them with an estimated dollar figure. A provider that does
not report usage yields zeros instead of an exception, so a run against an
unpriced backend still reports what it knows.

---

## Layout

```
src/ai_pair_engineer/
├── agents/
│   ├── base.py        # shared stage machinery, context budgeting, usage
│   ├── analyzer.py    # stage 1
│   ├── tester.py      # stage 2
│   ├── refactor.py    # stage 3
│   └── reviewer.py    # stage 4
├── models/schemas.py  # Pydantic contracts + severity weights
├── services/llm.py    # client, retries, JSON recovery, token accounting
├── static/
│   └── python_analyzer.py   # AST analysis, no API key needed
├── prompts/           # packaged, loaded via importlib.resources
├── pipeline.py        # orchestration
└── cli.py             # command line interface

benchmarks/
├── models.py          # PlantedDefect, DetectionCase, RegressionCase, Tier
├── corpus.py          # loads cases from disk, validates annotations
├── matching.py        # finding -> defect, by line or by keyword groups
├── scoring.py         # recall, groundedness, clean noise, breakdowns
├── runner.py          # tier execution, caching, cost accounting
├── report.py          # Markdown and JSON
├── validate.py        # corpus integrity checks, run in CI
└── cases/             # the corpus
```

Prompts are package data loaded through `importlib.resources`, so they resolve
regardless of the working directory. `benchmarks/` sits outside `src/` and is not
in the wheel: it is a tool for this repository and ships with the corpus it
measures.

---

## Limitations

Worth being explicit about, since "AI reviewer" usually overstates what these
tools do.

- **One file at a time.** No cross-file or repository-level context, so it
  cannot see that a helper is defined in a sibling module.
- **Static analysis is Python-only.** Other languages get the LLM stages but no
  local evidence. The report says `not available for this language` rather than
  pretending.
- **Security findings are pattern-based.** No dataflow analysis, so taint
  tracking and injection reachability are out of reach.
- **Generated tests are never run.** They are a starting point, not proof.
- **Refactors are not verified by execution.** The reviewer compares the two
  versions by reading them. It is good at catching inverted conditions and
  dropped branches, and it is not a proof.
- **Cost scales with file size.** Four calls per file. On a very large file the
  analyzer prompt is truncated at 20,000 characters.

---

## Roadmap

- [ ] Diff-aware review: compare against a base branch, not the whole file
- [ ] Multi-file review with an import graph
- [ ] Sandboxed test execution with resource limits
- [ ] SARIF output for GitHub code scanning
- [ ] AST-aware refactoring verification
- [ ] **An independent corpus.** The current one was written by the same person
      who wrote the analyzer, which is the benchmark's weakest point and the one
      an interviewer would press on first.

The benchmark harness exists. The corpus being self-authored is the limitation
it has not solved.

---

## License

MIT. See [LICENSE](LICENSE).
