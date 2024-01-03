FROM python:3.11

WORKDIR /gleam

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

COPY pyproject.toml /gleam/pyproject.toml

RUN poetry config virtualenvs.create false
RUN poetry install

ENTRYPOINT ["/bin/bash"]