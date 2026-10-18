<!-- SPDX-FileCopyrightText: 2026 Jan Owoc -->
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Maintaining the pins

Every version the image and CI install is pinned, so that two builds of one
commit install the same packages on the same base image. GitHub's runner
image, `ubuntu-24.04`, is not pinned, and neither are the system libraries
that `playwright install --with-deps` installs on it. This page lists each
pin and how it moves.

| Pin | Where | Updated by |
| --- | --- | --- |
| Python packages, runtime and dev | `uv.lock`, resolved from `pyproject.toml` | Dependabot (`uv`) |
| Python base image, tag and digest | `Dockerfile`, the `python` `FROM` | Dependabot (`docker`) |
| Python patch version | `.python-version` | by hand, with the base image |
| uv | `Dockerfile`, the `uv` stage | Dependabot (`docker`) |
| uv in CI | `setup-uv` `version` in both jobs of the workflow | by hand, with the `uv` stage |
| GitHub Actions | the workflow's `uses:` commits | Dependabot (`github-actions`) |
| setuptools, the build backend | `[tool.uv] build-constraint-dependencies` in `pyproject.toml` | by hand, with `uv.lock` |
| ruff | `pyproject.toml` and `uv.lock` | by hand, in a commit of its own that reformats the tree |
| Pyodide | `PYODIDE_VERSION` in `browser/worker.js`, and `tests/browser/package.json` with `package-lock.json` | by hand, all three together; `tests/browser/test_browser_page.py` fails when they differ. Dependabot does not watch npm |
| Node in CI | `setup-node` `node-version` in the workflow | by hand |
| Playwright and its Chromium | `playwright` in `uv.lock`; the Chromium build is the one that Playwright release names | Dependabot (`uv`). CI installs the matching Chromium on every run; a checkout runs `playwright install --only-shell chromium` again after an update |
| Alpine.js and Plotly | the two `<script>` tags in `browser/index.html`, and `tests/browser/package.json` with `package-lock.json` | by hand, together with `api/web/index.html`; `tests/browser/test_browser_page.py` fails when they differ. Dependabot does not watch npm |
| QEMU (binfmt), buildx, BuildKit | the image job's `with:` inputs | by hand, below |

`tests/test_pins.py` fails when a pin updated by hand has not followed the one
it goes with, and when a `FROM` or an action is not pinned to a digest or a
commit.

## Dependabot

`.github/dependabot.yml` opens at most one pull request per ecosystem a month,
grouping every update in it, and proposes a release only once it is seven days
old, when the registry gives the release's date. For `uv` it changes only
`uv.lock`; the lower bounds in `pyproject.toml` stay as they are, and ruff is
left alone. For `docker` it stays on Python 3.12: moving to a new minor version
is an issue of its own. Dependabot security updates, a repository setting, open
pull requests outside the monthly schedule and the seven-day wait.

The verifier agent reviews every Dependabot pull request before it is merged.
When the pull request's CI fails at `tests/test_pins.py`, the failure names the
pin to bring along: `.python-version` after a base-image patch, the `setup-uv`
versions after a uv bump, the build constraint after a setuptools bump. That
change is committed to the pull request's branch. A new Python patch can be
newer than the pinned uv knows how to download; CI then fails at `uv sync`, not
at `tests/test_pins.py`, and the `uv` stage and the `setup-uv` versions move in
the same pull request.

ghcr.io, where the uv image lives, gives Dependabot no release date, so the
seven-day wait does not hold for the `uv` stage: Dependabot proposes a uv
release however recent it is, and says at the top of the pull request that the
cooldown could not be applied. Whenever a pull request says so, the review
reads each release's date from the release notes in the pull request, and the
pull request is merged only once every release in it is seven days old.

## By hand

### QEMU, buildx and BuildKit

Checked on the first of each month. For each, take the newest release that is
at least seven days old:

- QEMU: a `qemu-v…` tag of `docker.io/tonistiigi/binfmt`.
- buildx: a release of `docker/buildx`, set as the `version` input.
- BuildKit: a release of `moby/buildkit`, the image in `driver-opts`.

Look up an image's digest with
`docker buildx imagetools inspect <image>:<tag> --format '{{json .Manifest.Digest}}'`
and write it after the tag. The pull request's CI builds both architectures
with the new pins.

### A new lock

`uv lock` resolves the newest versions. To keep the seven-day wait when
resolving from scratch, lock with a cut-off and then once without; the second
run keeps every version and leaves no cut-off in the file:

    uv lock --exclude-newer <the date seven days ago>T00:00:00Z
    uv lock
