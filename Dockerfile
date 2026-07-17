# syntax=docker/dockerfile:1.6
ARG GRADER_PY_IMAGE=python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203
FROM ${GRADER_PY_IMAGE} AS grader-rootfs

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ARG RELIQUARY_BUILD_REVISION=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RELIQUARY_BUILD_REVISION=${RELIQUARY_BUILD_REVISION}

RUN apt-get update -qq && apt-get install -y -qq \
        python3.12 python3.12-venv python3-pip \
        git build-essential wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so system pip stays clean
RUN python3.12 -m venv /opt/reliquary-venv
ENV PATH="/opt/reliquary-venv/bin:${PATH}"

# torch 2.7.0 + CUDA 12.8 (matches our Targon setup)
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn prebuilt wheel for torch 2.7 / cu12 / cp312 / cxx11abi=TRUE.
# Do NOT rename the wheel on download: pip parses the version and platform
# tags from the filename and rejects anything that doesn't match the
# ``name-version-...-abi.whl`` shape with "Invalid wheel filename
# (wrong number of parts)".
ARG FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN wget -q "${FA_URL}" -P /tmp/ \
 && pip install /tmp/flash_attn-*.whl \
 && rm /tmp/flash_attn-*.whl

# flash-linear-attention: fast triton kernel for Qwen3.5's GatedDeltaNet layers.
# Without it they fall back to the slow torch_chunk_gated_delta_rule scan (~3x
# slower GRAIL verify). causal-conv1d is skipped on purpose: it needs an nvcc
# compile and only swaps a cheap conv -- the gated-delta scan is the real cost.
RUN pip install flash-linear-attention==0.5.0

# Source + install
WORKDIR /opt/reliquary
COPY . /opt/reliquary
RUN pip install -e .

# boto3 for R2 (weight-only mode + trainer archive uploads)
RUN pip install boto3

# wandb for trainer telemetry (lazy-imported in reliquary.validator.telemetry).
# No-op at runtime if WANDB_API_KEY is unset.
RUN pip install wandb

# ────────────────  GRADER SANDBOX  ────────────────
# Install gVisor (runsc) for the OpenCodeInstruct env's sandbox.
# Pinned to an exact release + verified checksum: the grader's pass/total
# feeds rewards, so a different runsc across validators would break the
# cross-box-determinism guarantee (and ``latest`` changed every rebuild).
# Bump RUNSC_RELEASE deliberately.
ARG RUNSC_RELEASE=20260525
RUN ARCH="$(uname -m)" \
 && BASE="https://storage.googleapis.com/gvisor/releases/release/${RUNSC_RELEASE}/${ARCH}" \
 && wget -q "${BASE}/runsc" "${BASE}/runsc.sha512" \
 && sha512sum -c runsc.sha512 \
 && chmod +x runsc \
 && mv runsc /usr/local/bin/runsc \
 && rm -f runsc.sha512

# Build the grader OCI bundle (python:3.12-slim rootfs + worker.py).
# This is deterministic and fail-closed: the published image must contain a
# runnable rootfs, so OpenCode startup never depends on a Docker socket inside
# the validator container.
COPY scripts/build_grader_bundle.sh /opt/build_grader_bundle.sh
RUN chmod +x /opt/build_grader_bundle.sh
COPY --from=grader-rootfs / /opt/reliquary/reliquary/environment/grader/bundle/rootfs/
RUN install -m 0644 /opt/reliquary/reliquary/environment/grader/worker.py \
      /opt/reliquary/reliquary/environment/grader/bundle/rootfs/opt/worker.py \
 && test -x /opt/reliquary/reliquary/environment/grader/bundle/rootfs/usr/local/bin/python3

# Create the optional validator service user and a reserved grader group id.
# The trusted grader supervisor runs as root so runsc can create cgroups; the
# untrusted worker itself runs as UID/GID 65534 inside the OCI config.
RUN if ! getent passwd 1000 >/dev/null; then useradd -m -u 1000 reliquary; fi \
 && if ! getent passwd 1001 >/dev/null; then useradd -m -u 1001 reliquary-grader; fi

# Runtime
ENV GRAIL_ATTN_IMPL=flash_attention_2
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/opt/entrypoint.sh"]
