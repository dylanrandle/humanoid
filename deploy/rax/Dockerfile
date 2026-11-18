FROM ghcr.io/astral-sh/uv:0.11.32 AS uv
FROM node:22.21.1-bookworm-slim AS node
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap ca-certificates git procps ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && node /usr/local/lib/node_modules/npm/bin/npm-cli.js install -g @openai/codex@0.169.0 \
    && codex_binary="$(rg --files /usr/local/lib/node_modules/@openai | rg '/vendor/.*/bin/codex$')" \
    && ln -s "$codex_binary" /usr/local/bin/codex-linux-sandbox
COPY requirements.txt /tmp/rax-requirements.txt
RUN uv venv --python /usr/local/bin/python3.12 /opt/rax-python \
    && uv pip install --python /opt/rax-python/bin/python -r /tmp/rax-requirements.txt \
    && printf '%s\n' 'export PATH=/opt/rax-python/bin:$PATH' > /etc/profile.d/rax-python.sh
RUN useradd --create-home --uid 1000 rax \
    && mkdir -p /home/rax/runtime /home/rax/.codex \
    && chown -R rax:rax /home/rax
COPY --chown=rax:rax rax /home/rax/source
RUN ln -s /home/rax/runtime /home/rax/source/runtime
USER rax
WORKDIR /home/rax/source
ENV RAX_ARTIFACT_ROOT=/home/rax/runtime PYTHONDONTWRITEBYTECODE=1 UV_NO_CACHE=1 PATH=/opt/rax-python/bin:$PATH
ENTRYPOINT ["uv", "run", "--no-project", "--python", "/opt/rax-python/bin/python", "python"]
CMD ["-m", "harness.codex_cli.auto_research", "--help"]
