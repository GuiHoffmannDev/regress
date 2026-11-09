FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY regress ./regress
RUN pip install .

COPY prompts ./prompts
COPY golden ./golden

# Run history (.regress/) and reports (out/) are written here; mount volumes to keep them.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/.regress /app/out \
    && chown -R app:app /app/.regress /app/out
USER app

# Ollama on the host: OPENAI_BASE_URL=http://host.docker.internal:11434/v1 OPENAI_API_KEY=ollama
# Configure with env vars: OPENAI_API_KEY, OPENAI_BASE_URL, JUDGE_MODEL, JUDGE_REASONING_EFFORT, SLACK_WEBHOOK_URL,
# REPORT_URL, REGRESS_WARN_DELTA, REGRESS_CRITICAL_DELTA, REGRESS_DRIFT_FLOOR,
# REGRESS_DRIFT_WINDOW, REGRESS_MIN_SUMMARY_SCORE, REGRESS_CONCURRENCY.
ENTRYPOINT ["regress"]
CMD ["run"]
