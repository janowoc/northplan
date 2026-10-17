# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

# A BuildKit parser directive — `# syntax=...` — would have to go ABOVE the two
# lines here. BuildKit stops looking for directives at the first comment, blank
# line, or instruction, so one placed below them is read as an ordinary comment
# and silently ignored: no error, just the default frontend. There is no such
# directive today.

# uv installs the locked packages. It is a stage of its own so that its pin is a
# FROM line, and it is mounted for the one RUN that uses it, so it is not in the
# image.
FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

# One container: the API and the static files it serves.
FROM python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY engine/ engine/
COPY api/ api/
COPY cli/ cli/
COPY report/ report/
COPY params/ params/
COPY scenarios/example.yaml scenarios/

# The versions in uv.lock, into the image's own Python. The project goes in
# editable, so that the installed code is /app's: the `northplan` command and
# the server both default to /app/params, and the server to
# /app/scenarios/example.yaml, the paths compose mounts over. A plain install
# would put a second copy of the code in site-packages, where neither exists.
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    UV_PROJECT_ENVIRONMENT=/usr/local UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    uv sync --locked --no-cache

# The image carries params/ and scenarios/example.yaml (what /api/example
# serves), so it runs without a checkout. docker-compose bind-mounts ./params
# and ./scenarios over them: hand-edited files take effect on the next
# request, without a rebuild.
EXPOSE 8000

# northplan-serve runs uvicorn with proxy headers off, so the server sees each
# connection's real peer. Unless NORTHPLAN_TOKEN is set it generates a token on
# each start and writes it to the log; every client of a container needs it.
CMD ["northplan-serve", "--host", "0.0.0.0"]
