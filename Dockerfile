FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
EXPOSE 7860
CMD ["uv", "run", "server"]
