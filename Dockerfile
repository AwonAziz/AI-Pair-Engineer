# Pinned for reproducible builds. Bump deliberately, not on every patch.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copied separately so dependency installation is cached independently of
# application source changes.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

COPY app.py ./
COPY examples ./examples

# Run as a non-root user. The container analyses untrusted source, so it should
# not have write access to the image.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
