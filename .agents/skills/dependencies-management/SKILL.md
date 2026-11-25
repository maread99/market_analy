---
  name: dependencies-management
  description: instructions for updating and managing project dependencies (including Github actions).
---
# Dependencies Management

## Update Dependencies

Instructions to update project dependencies (including Github actions used by CI workflows).

### 1. Prepare the branch

Create a new branch (following the naming convention in @AGENTS.md).

### 2. Update the lock file and environment

Update `uv`:
```bash
uv self update --token $GITHUB_TOKEN
```

**IMPORTANT: updating `uv` MUST NOT be skipped.** If the above command fails (e.g. due to a GitHub API rate limit), try the following fallbacks in order until `uv --version` confirms the latest version is active on PATH:

**Fallback 1 — official install script** (preferred; replaces PATH binary directly):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # refresh current shell session
uv --version                 # verify PATH binary is now updated
```

**Fallback 2 — pip** (use only if `curl` is unavailable):

```bash
pip install --user --upgrade uv
uv --version   # check if PATH binary was updated
```

Once `uv` is up to date, run:

```bash
uv lock --upgrade  # update the lock file
uv export --format requirements-txt --no-emit-project --no-hashes --no-dev -o requirements.txt  # sync @requirements.txt with @uv.lock
uv sync --inexact  # update environment to match @uv.lock
```

### 3. Update GitHub Actions versions

For each `uses: <owner>/<repo>@<version>` entry across all files in `.github/workflows/`, list the repository's tags with the newest version first:

```bash
git ls-remote --tags --refs --sort=-v:refname https://github.com/<owner>/<repo>
```

In each case update the version pin to the tag of any more recent stable release (ignore the tag of any pre-release, for example `v8.0.0-rc.1`). **Preserve the existing pinning style**, so if the current pin is a specific version such as `@v3.0.1`, update to the full latest version string, whilst if it's a major-version tag such as `@v4`, update to any new major-version tag (if the project no longer publishes a major-version tag then switch to the full semver). Note that the listing places a major-version tag (for example `v4`) after every full version tag of that major version.

### 4. Update pre-commit hook versions

Update the `rev:` pin of each repository in `.pre-commit-config.yaml` to the repository's latest tag:

```bash
uv run pre-commit autoupdate
```

Then run the hooks against all files:

```bash
uv run pre-commit run --all-files
```

Keep any changes that the hooks make. If a hook reports a violation that the hook cannot fix, fix the violation and run the hooks again.

### 5. Test

Run the test suite:
```bash
uv run pytest -v
```

**Interpret local test results:**
- All tests pass and no raised warning is fixable (see *Fixable warnings* section below) → go to step 7 to raise PR.
- Every test failure or error is a network test failure (see *Network tests* section below) and no raised warning is fixable → go to step 7 to raise PR. The CI checks will verify the network tests (see step 9).
- Any other test failure or error, or a fixable warning raised → proceed to step 6 to fix.

### 6. Fix
Any failing tests and fixable warnings will likely have their origin in changes to the dependencies. To provide support for the latest dependencies MAKE REVISIONS to the code base to fix:
- code causing tests to fail.
- all fixable warnings (see *Fixable warnings* section below).

As a general RULE, **change the package code to get the tests passing, not the test code!** You may make changes to the test code only with good reason and only when this does not impair the test's efficacy.

To facilitate identifying the cause of test failures consider researching the changelogs of updated dependencies for versions released since the previously locked version.

Iterate on this process until:
- all tests are passing with the exception of any network test failures (see *Network tests* section below).
- all fixable warnings have been fixed.

IMPORTANT: in this step you should not run the full test suite, rather validate fixes by re-running only the previously failing tests. Example to run a specific test:

```bash
pytest tests/test_module.py::test_name
```

### 7. Raise PR

Once local tests pass, commit all changes to the branch and raise a PR.

- **PR title**: Title the PR as `Update Dependencies <MM> <DD> (auto)` where:
  - `<MM>` should be replaced with the first three letters of the current month, the first of which should be capitalized.
  - `<DD>` should be replaced with the current day of the month as represented by two digits.
  Example title: `Update Dependencies Apr 07 (auto)`
- **label**: Add the 'dependencies' label to the PR.
- otherwise comply with the package's 'create-pr' skill.

### 8. Subscribe to PR activity

Raising the PR will trigger GitHub CI workflows. Subscribe to the raised PR's activity.

Proceed to step 9 when you receive events indicating that all triggered workflows have completed. A push to the PR cancels any workflow run in progress for an earlier push. An event for a cancelled workflow run of an earlier push does not indicate completion. Continue to wait for the workflow runs of the latest push.

### 9. Inspect CI results

The CI workflow will have run the full test suite against a matrix of OS and Python versions. Use `mcp__github__pull_request_read` with the `get_check_runs` method to read check statuses for all matrix jobs.

If any check is red, read the logs of the failed jobs to identify each failing test and the cause of its failure. To read the logs, call `mcp__github__get_job_logs` with the `run_id` of the failed workflow run, `failed_only: true` and `return_content: true`. If a returned log does not reach back to the test failures, call again with a larger `tail_lines`.

The CI checks are the arbiter of the network test failures of the local test run (see *Network tests* section below). A network test that passes in CI needs no fix.

**Interpret CI test results:**
- All checks green → proceed to step 12.
- Every test failure or error is a network test failure → add a comment to the PR that identifies the failures as possible network issues and asks the owner to re-run the failed jobs. Suspend the session pending further prompting.
- Any other check red → proceed to step 10.

### 10. Fix tests for specific OS/Python configuration

Use the information read from `get_check_runs` to identify any OS/python version configurations for which the test suite has failed.

If the tests failed on specific matrix combinations (e.g. Windows / Python 3.10) then simulate a local matching environment.
- If necessary use a Docker container with the target OS. (To specify different Python versions you will be able to run commands with `uv run` and pass the --python option.)
- create a new branch against the repository's main branch.
- update the `uv` package with `uv self update`.
- overwrite `uv.lock` with the version previously created by step 2.
- synchronise the environment by running `uv sync`.

Run the test suite to identify failing tests. For example:

```bash
uv run --isolated --python 3.11 pytest -v
```

If the local test environment cannot reach the Yahoo Finance API, the network tests cannot be verified locally (see *Network tests* section below). In that case, find the cause of a network test's failure from the CI logs read in step 9.

Then find fixes for the failing tests by following step 6.

Finally commit the necessary changes to your *original* branch (to which previous commits were made). This will trigger the CI on the PR re-run. Return to step 9 (inspect CI results).

### 11. Fallback: raise an issue

ONLY if any test failures cannot be resolved, raise an issue that references the PR and details:
- the failing tests
- any fixes already attempted
- any suggested next steps.

### 12. Tidy up

Perform the following 'tidy up' actions:
- Call `mcp__github__unsubscribe_pr_activity` to unsubscribe from the PR activity.
- Review and if necessary update the PR body to reflect the final circumstances.

---

**Network tests** — in `tests/test_analysis.py` and `tests/test_guis.py`

Many tests in `tests/test_analysis.py` and `tests/test_guis.py` require live network access to the Yahoo Finance API. Each such test constructs a `PricesYahoo` instance, either directly or via a fixture of `tests/conftest.py` that reconstructs the instance from pickled prices (see `_unpickle_prices`). Constructing the instance requests data from the Yahoo Finance API. If the API refuses the request (for example with HTTP 429 Too Many Requests), the test fails or errors, typically with a `TypeError` raised from `market_prices/prices/yahoo.py`.

A test failure or error is a *network test failure* if its traceback passes through the construction of a `PricesYahoo` instance.

The Yahoo Finance API can refuse requests from the local test environment, in which case the network tests cannot be verified locally. The CI checks verify the network tests (see step 9).

---

**Fixable warnings**
The following warnings are considered UNFIXABLE and you should not attempt to fix them.
- warnings that have their origin in a dependency's code.
- `PricesMissingWarning`.

All other warnings are considered fixable.

## Adding dependencies
```bash
# Add a dependency
uv add <package>
```
