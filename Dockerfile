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
COPY web/ web/
COPY params/ params/

RUN pip install --upgrade pip && pip install .

# params/ and scenarios/ are bind-mounted by docker-compose so that
# hand-edited parameters and local scenarios take effect without a rebuild.
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
