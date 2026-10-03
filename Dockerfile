# homeduplex server image.
#   docker build -t homeduplex .
#   docker run -p 8770:8770 -v ./homeduplex.yaml:/config/homeduplex.yaml:ro homeduplex
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --system --no-create-home homeduplex
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/LICENSE /app/NOTICE /app/THIRD_PARTY_NOTICES.md /usr/share/doc/homeduplex/
ENV PATH=/app/.venv/bin:$PATH \
    HOMEDUPLEX_CONFIG=/config/homeduplex.yaml \
    HOMEDUPLEX_SERVER__HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1
USER homeduplex
EXPOSE 8770
# Assumes the default port; change it here too if server.port is changed.
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/healthz', timeout=3)"
ENTRYPOINT ["homeduplex"]
CMD ["serve"]
