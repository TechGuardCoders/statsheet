# statsheet — statistical significance engine container
# Build:  docker build -t statsheet .
# Run:    docker run statsheet                       # usage
#         docker run -v $PWD:/data statsheet \
#           compare /data/a.jsonl /data/b.jsonl

FROM python:3.12-slim AS builder
WORKDIR /app
COPY statsheet.py .

FROM python:3.12-slim AS runner
WORKDIR /app
COPY --from=builder /app/statsheet.py .
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "statsheet.py"]
CMD ["--help"]
