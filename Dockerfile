FROM python:3.11 AS signals-classification-base

ENV PYTHONUNBUFFERED 1

RUN set -eux; \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3; \
    cd /usr/local/bin; \
    ln -s /opt/poetry/bin/poetry; \
    poetry config virtualenvs.create false; \
    poetry completions bash >> ~/.bash_completion

COPY pyproject.toml poetry.lock* /app/

WORKDIR /app

RUN poetry install --no-root --with train,web

COPY . /app

FROM signals-classification-base AS signals-classification-web

WORKDIR /app/app

ENV UWSGI_HTTP :8000
ENV UWSGI_MODULE app:application
ENV UWSGI_PROCESSES 8
ENV UWSGI_MASTER 1
ENV UWSGI_OFFLOAD_THREADS 1
ENV UWSGI_HARAKIRI 25

CMD uwsgi

FROM signals-classification-base AS signals-classification-train

RUN mkdir /tmp/nltk

ENV NLTK_DATA /tmp/nltk
ENV PYTHONPATH=/app:$PYTHONPATH

WORKDIR /app

ENTRYPOINT ["python", "-m", "app.train.run"]