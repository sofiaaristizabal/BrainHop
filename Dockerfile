FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for psycopg binary and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==2.3.4

# Copy dependency files first (better Docker layer caching)
COPY pyproject.toml poetry.lock ./

# Configure Poetry: don't create a virtualenv inside the container,
# install directly into the system Python
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application
COPY . .

# Install the app itself (so `app` package is importable)
RUN poetry install --no-interaction --no-ansi

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app.main:app"]

CMD ["--host", "0.0.0.0", "--port", "8000"]