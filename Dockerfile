# ---- estágio 1: build ----
FROM python:3.12-slim AS build
WORKDIR /build
ENV POETRY_VIRTUALENVS_CREATE=false
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

# ---- estágio 2: runtime ----
FROM python:3.12-slim AS runtime
RUN addgroup --system recsys && adduser --system --ingroup recsys recsys
WORKDIR /app
RUN mkdir -p /app/mlruns
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY src/ ./src
ENV PYTHONPATH=/app/src
COPY scripts/ ./scripts
COPY configs/ ./configs
COPY params.yaml ./
RUN chown -R recsys:recsys /app
USER recsys
ENTRYPOINT ["python"]
