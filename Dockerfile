FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY desic ./desic
RUN pip install --no-cache-dir .
ENV DESIC_DATA=/data
VOLUME /data
EXPOSE 8000
CMD ["desic", "serve", "--host", "0.0.0.0", "--port", "8000", "--data", "/data"]
