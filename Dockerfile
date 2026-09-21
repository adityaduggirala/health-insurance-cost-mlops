FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -e . && mkdir -p artifacts && python -m insurance.train
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "insurance.api:app", "--host", "0.0.0.0", "--port", "8000"]
