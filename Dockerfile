FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin morpheus

COPY pyproject.toml README.md ./
COPY morpheus ./morpheus
COPY ui ./ui

RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && chown -R morpheus:morpheus /app

USER morpheus

EXPOSE 8000 5173

CMD ["morpheus", "serve", "--ui", "--host", "0.0.0.0", "--port", "8000", "--ui-host", "0.0.0.0", "--ui-port", "5173", "--ui-root", "/app"]
