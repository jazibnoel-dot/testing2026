# ============================================================
#  Multi-stage Dockerfile for NoShowIQ
#  Base image : python:3.11-slim   (required by brief)
#  Final image: < 300 MB, runs as non-root user "app"
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only the compiler tools we need during the build.
# These do NOT copy into the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# --user installs into ~/.local so we can copy it cleanly later
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user (security best practice + brief requirement)
RUN useradd --create-home --shell /bin/bash app

WORKDIR /home/app

# Copy only the installed packages from the builder stage
COPY --from=builder /root/.local /home/app/.local

ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy application source and saved model
COPY --chown=app:app noshow_iq/ ./noshow_iq/
COPY --chown=app:app models/    ./models/

USER app

# Hugging Face Spaces routes traffic to port 7860 by default.
# Do NOT change this to 8000 or the Space will 502 forever.
EXPOSE 7860

CMD ["uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "7860"]
