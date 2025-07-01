FROM python:3.10.6

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY aml aml


CMD uvicorn aml.api.fast:app --host 0.0.0.0 --port $PORT
# CMD ["uvicorn", "aml.api.fast:app", "--host", "0.0.0.0", "--port", "$PORT"]
