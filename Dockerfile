# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

# A BuildKit parser directive — `# syntax=...` — would have to go ABOVE the two
# lines here. BuildKit stops looking for directives at the first comment, blank
# line, or instruction, so one placed below them is read as an ordinary comment
# and silently ignored: no error, just the default frontend. There is no such
# directive today.

# One container: the API and the static files it serves.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY engine/ engine/
COPY api/ api/
COPY cli/ cli/
COPY report/ report/
COPY web/ web/
COPY params/ params/
COPY scenarios/example.yaml scenarios/

RUN pip install --upgrade pip && pip install .

# The image carries params/ and scenarios/example.yaml (what /api/example
# serves), so it runs without a checkout. docker-compose bind-mounts ./params
# and ./scenarios over them: hand-edited files take effect on the next
# request, without a rebuild.
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
