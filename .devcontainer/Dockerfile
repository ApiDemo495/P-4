# =============================================================================
# Pre-provisioned dev container for DROSOPHILA TRADER v2.0
#
# Built by .github/workflows/devcontainer-image.yml and pushed to
#   ghcr.io/apidemo495/p-4-dev:latest
# so a new Codespace pulls one cached image instead of running apt + pip.
# Everything requirements.txt needs is already installed in /opt/venv, which
# the repo scripts adopt as .venv (symlink) on first start - zero downloads.
# =============================================================================
FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# system tools the scripts may call (redis is optional for the engine; gh is
# used to publish the port)
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
        redis-server curl git unzip xz-utils ca-certificates gnupg \
 && (curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
     && apt-get update -qq && apt-get install -y -qq --no-install-recommends gh) \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# the Python environment, fully resolved at image-build time
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --quiet --upgrade pip wheel setuptools \
 && /opt/venv/bin/pip install --quiet --prefer-binary -r /tmp/requirements.txt \
        websockets pyflakes markdown \
 && sha256sum /tmp/requirements.txt | cut -d' ' -f1 > /opt/venv/.requirements.sha256 \
 && chown -R vscode:vscode /opt/venv

# ---------------------------------------------------------------------------
# Round AV: everything a fresh Codespace used to download or that the user
# used to upload is baked in here, once, at image-build time:
#   * the GGUF runtime (prebuilt CPU wheel; compiled only if no wheel matches)
#   * the FlyWire v783 connectome (parquet + annotations + the built .npz)
#   * every enabled model in models.json (1 GB-class weights)
# run.sh links /opt/drosophila/{connectome,models} into the checkout.
# ---------------------------------------------------------------------------
RUN ( /opt/venv/bin/pip install --quiet --prefer-binary \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python \
    || ( apt-get update -qq && apt-get install -y -qq --no-install-recommends cmake build-essential \
         && CMAKE_ARGS="-DGGML_NATIVE=OFF" /opt/venv/bin/pip install --quiet llama-cpp-python \
         && apt-get clean && rm -rf /var/lib/apt/lists/* ) ) \
 && /opt/venv/bin/python -c "import llama_cpp; print('llama_cpp', llama_cpp.__version__)"

COPY backend /tmp/src/backend
COPY tools/fetch_models.py /tmp/src/tools/fetch_models.py
COPY models.json /tmp/src/models.json
RUN mkdir -p /opt/drosophila/connectome /opt/drosophila/models \
 && cd /tmp/src \
 && PYTHONPATH=/tmp/src CONNECTOME_DATA_DIR=/opt/drosophila/connectome \
      /opt/venv/bin/python -c "from backend.brain.connectome_data import ConnectomeData; d = ConnectomeData(); assert d.ensure(), d.status.error; print(d.facts)" \
 && PYTHONPATH=/tmp/src /opt/venv/bin/python tools/fetch_models.py --dest /opt/drosophila/models --quiet \
 && rm -rf /tmp/src \
 && chown -R vscode:vscode /opt/drosophila \
 && du -sh /opt/drosophila/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.source="https://github.com/ApiDemo495/P-4"
LABEL org.opencontainers.image.description="DROSOPHILA TRADER v2.0 dev container: Python 3.11 + every requirement pre-installed"
