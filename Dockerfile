FROM python:3.12

WORKDIR /app
RUN python -m venv /venv
ENV PATH="$PATH:/venv/bin:$PATH"

COPY . .
RUN pip install .

ENTRYPOINT [ "physics-sim" ]