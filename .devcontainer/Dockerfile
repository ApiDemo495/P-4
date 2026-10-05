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
# used to publish the port; unzip/xz only matter for the Flutter fallback)
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

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.source="https://github.com/ApiDemo495/P-4"
LABEL org.opencontainers.image.description="DROSOPHILA TRADER v2.0 dev container: Python 3.11 + every requirement pre-installed"
