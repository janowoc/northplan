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

Dependabot may fail to update setuptools, in a security update or a scheduled
one: the security update for GHSA-xmj3-9gf7-h52w failed in its job with `No
files have changed!`, for a reason not known, and opened no pull request. The
update is then made by hand, moving the build constraint and the lock
together: set `build-constraint-dependencies` in `pyproject.toml` to
`["setuptools==<version>"]`, and run
`uv lock --upgrade-package setuptools==<version>` with the same version. The
build constraint does not steer the lock:
`uv lock --upgrade-package setuptools` alone takes the newest release, a plain
`uv lock` keeps the old one, and `tests/test_pins.py` fails whenever the build
constraint and the lock differ.

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

## Releasing

The version is written once, as `version` in `pyproject.toml`. A release is
a commit and then a tag:

1. Set that version, run `uv lock`, and commit `pyproject.toml` and `uv.lock`
   as `chore(release): X.Y.Z`. The lock changes only northplan's own entry.
2. Push it to main, and wait until the run for that push has passed, its
   deploy of the page included (`gh run watch --exit-status` follows it).
3. Then, with no other run of main or of a `v*` tag in progress or queued,
   tag that commit with an annotated tag and push the tag. Until the tag's run
   has finished, push nothing else to main or as a `v*` tag, and re-run no run.
   A deploy fetches every tag when its pages job checks out, so a tag that
   exists then could be served by the root before its own run had tested it:

       git tag -a vX.Y.Z -m "northplan X.Y.Z"
       git push origin vX.Y.Z

   The test job refuses a tag that is not `v` followed by the version in
   `pyproject.toml`. The run publishes the image as `X.Y.Z`, `X.Y` and
   `latest`, and deploys the page. The `github-pages` environment allows
   deployments from `v*` tags as well as from main.
4. Check that the image and the page both report the new version. The status
   block of https://janowoc.github.io/northplan/ reads `northplan X.Y.Z`, and
   this prints `X.Y.Z`:

       docker run --rm --pull always ghcr.io/janowoc/northplan:latest python -c "import importlib.metadata as m; print(m.version('northplan'))"

5. Create the release page on GitHub from the tag.

The page's root serves the annotated `vN.N.N` tag with the newest creation
date, and `main/` serves the head of main. The page ignores a lightweight tag
and every other `v*` name; the image still publishes a lightweight `vN.N.N`
tag, so tags must be annotated, as step 3 shows. A deploy fails, and does not
fall back to another tag, when two such tags share the newest creation date,
when the tag's `pyproject.toml` version is not the tag's, or when the tag's
tree has no `browser/build.py`.

If a tag's test job fails, nothing was published or deployed from it. Delete
the tag before anything else is pushed to main or as a `v*` tag, and before any
run is re-run: if it is the newest annotated `vN.N.N` tag, the next deploy
would otherwise build the root from it, or fail on it. Delete it with:

    git push origin --delete refs/tags/vX.Y.Z
    git tag -d vX.Y.Z

If the test job passed and the image or pages job failed, do not delete the
tag: the other job may already have published or deployed from it. Re-run the
failed jobs:

    gh run rerun <run-id> --failed

The image's `latest` follows the `v*` tag whose run published the image most
recently, a different rule from the page's root. The two can pick different
tags: for example when tags are pushed out of the order they were created,
when two tags' runs overlap, when a tag's run fails, or when an older tag's
run is re-run. Create and push one tag at a time, each after the previous
tag's run has finished, and check after each release that the image and the
page both report the new version (step 4). A backport tagged after a newer
version, such as v0.2.1 after v0.3.0, becomes both the image's `latest` and the
page's root.
[#119](https://github.com/janowoc/northplan/issues/119) is to change that.
