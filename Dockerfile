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
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

# One container: the API and the static files it serves.
FROM python:3.12.15-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY engine/ engine/
COPY api/ api/
COPY cli/ cli/
COPY report/ report/
COPY params/ params/
COPY scenarios/example.yaml scenarios/minimal.yaml scenarios/

# The versions in uv.lock, into the image's own Python. The project goes in
# editable, so that the installed code is /app's: the `northplan` command and
# the server both default to /app/params, and the server to
# /app/scenarios/example.yaml and /app/scenarios/minimal.yaml, the paths
# compose mounts over. A plain install would put a second copy of the code in
# site-packages, where none of them exists.
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    UV_PROJECT_ENVIRONMENT=/usr/local UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    uv sync --locked --no-cache

# The image carries params/, scenarios/example.yaml and scenarios/minimal.yaml
# (what /api/example and /api/minimal serve), so it runs without a checkout.
# docker-compose bind-mounts ./params and ./scenarios over them: hand-edited
# files take effect on the next request, without a rebuild, except that the
# editor's calls keep the parameters they have loaded until a restart.
EXPOSE 8000

# northplan-serve runs uvicorn with proxy headers off, so the server sees each
# connection's real peer. Unless NORTHPLAN_TOKEN is set it generates a token on
# each start and writes it to the log; every client of a container needs it.
CMD ["northplan-serve", "--host", "0.0.0.0"]
