---
name: verify-test
description: This skill should be used whenever a test is written or edited in this repo — new checks in tools/check_logic.py, a new validator, a changed assertion. Trigger it on "add a test", "write a check", "pin this behaviour", or after any fix that should be guarded. A test that cannot fail is worse than no test, because it reports safety that is not there; this skill is the six checks that catch that, each of which caught a real defect here.
---

# Verify the Test Can Actually Fail

Write the test, then prove it fails without the fix. Every step below exists
because skipping it shipped a broken guard in this repo.

## Six checks, each earned

**1. BREAK THE FEATURE AND CONFIRM THE TEST FAILS.** Not optional. Verified
failures here:
- A test asserting the market regime drives the correlation matrix compared two
  single-asset books — each draws one number at the same stream position, so it
  passed against a deliberately per-ticker implementation.
- A threshold of `0.70` against a true per-ticker ratio of `1/sqrt(2) = 0.707`
  let the broken version through by a thousandth.
- A smoke test asserting "enough digits to look like a table" passed with
  `_flat_table` emptied — the exact silent-missing-table failure it existed for.

Break it in the source, re-run, restore. If it still passes, the test is theatre.

**2. COUNT THE TESTS BEFORE AND AFTER.** Replacing a function by slicing from its
name to a distant anchor silently deleted ELEVEN committed tests. It was caught
only because the pass line read 221 where it had read 234.

```
python3 -c "import re; print(len(re.findall(r'^def test_', open('tools/check_logic.py').read(), re.M)))"
git diff tools/check_logic.py | grep '^-def test_'    # must be empty
```

Bound a replacement by the next TOP-LEVEL LINE OF ANY KIND — the first
non-blank line that is not indented — never by a named anchor, and never by
the next `def`. Both of those deleted code in one session: `index('def ')`
matched a NESTED helper and left half a test behind as an IndentationError, and
`index('\ndef ')` jumped over the `_CONFIG_FINGERPRINT = ...` and `CHECKS = ...`
assignments between the last test and `def main`, so the suite did not shrink,
it CRASHED (`NameError: CHECKS`). Count what the runner needs, not just tests:

```
python3 - <<'X'
lines = open('tools/check_logic.py').read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith('def test_NAME'))
end = next(i for i in range(start + 1, len(lines))
           if lines[i].strip() and not lines[i][0].isspace())
print(lines[end])            # must be the NEXT thing, whatever it is
X
```

**3. NEVER `importlib.reload` A MODULE THE ENGINE BINDS.** `engine` binds
`factors` at import and copies scalars, so reloading `factors` mid-suite leaves
`engine` holding stale values and makes later tests order-dependent. That made
`check_logic` fail intermittently under `validate_all` while passing standalone.
Use a subprocess — which is what production does anyway, since every `MC_` var is
read at import and workers re-import under `spawn`.

**4. REDIRECT ANYTHING THE TEST WRITES.** A timing test called
`instrument.summary()`, which calls `calibrate()`, which WRITES — with no
redirect. Every `validate_all` wrote a fake 9,000-path run into the user's
`config/timing.json`, dragging the stored constant toward zero and making the
run-time estimate progressively wrong. Two commits chased the symptom.
`test_zzz_the_suite_left_no_user_config_modified` now guards the directory, but
redirect to a tempfile in the first place.

Content equality is not enough to prove it: a preset test wrote the real
`config/runs.json` and then restored it, so the bytes matched and the guard
passed — while the cloud-synced folder turned the two writes into an
edit-conflict copy. The guard now compares mtimes too. And when you break-test
a redirect (rule 1), do it in a throwaway `git worktree` with `output/`
symlinked in: reinstating the write WRITES YOUR REAL FILE. It did, 2026-09-29,
and `config/runs.json` had to be repaired by hand.

**5. ASSERT ON CODE, NOT ON WORDS IN PROSE.** Twice in one session a test
grepped module source for a term that appears in a comment EXPLAINING why the
term is not used: `'random' not in src` failed against a docstring saying "not
drawn per path... a random draw would average the shortfall away", and
`'ROUND_TRIP' not in src` failed against "factors.ROUND_TRIP already prices
each FUND's internal turnover". Both tests were right about the code and wrong
about how to look at it. Strip docstrings first, or better, assert on
behaviour — render the output, call the function, compare the number.

**6. A TEST THAT DRIVES A PROMPT MUST RAISE ON EXHAUSTED INPUT.** Twice in one
session a scripted `ask` returned `''` (or its default) once its script ran out,
and a prompt that RE-ASKS on bad input looped forever: the whole gate hung for
ten minutes instead of failing, which reads as a slow test, not a defect. Worse,
`ask` returns its DEFAULT on EOF, so closing stdin does not help — a reinstated
confirm prompt silently auto-accepted. Script the answers and raise when they
run out:

```python
def _scripted(keys):
    it = iter(keys)
    def _ask(_p, _d=''):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError('the prompt asked for more input than scripted')
    return _ask
```

And when a prompt must NOT be reached at all, script `[]` — any call fails.

**A break-it check (rule 1) must show the suite RAN.** A mutation loop once
wrapped the suite in `timeout`, which macOS does not have: every run failed to
start, `|| true` swallowed it, and "no FAIL lines" looked exactly like "every
mutation survived" — or, read the other way, like a clean pass. Grep for the
summary line too (`All N pure-logic checks passed` / `N/M checks failed`), so
a run that never happened cannot be mistaken for one that found nothing.

## Also

- Assert the INVARIANT, not the formula. A test pinning
  `0.5 * measured + 0.5 * prior` broke the moment the blend weights moved, for a
  reason unrelated to what it guarded.
- Do not reword a message a test pins unless the rewording is the point. Twice
  in one session a gratuitous rename broke a working test; the fix both times
  was to stop making the change.

## Reference

- Conventions and fixtures: `tools/check_logic.py`
- The gate: `tools/validate_all.py`
