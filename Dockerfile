# syntax=docker/dockerfile:1.6
ARG GRADER_PY_IMAGE=python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203
FROM ${GRADER_PY_IMAGE} AS grader-rootfs

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG RELIQUARY_BUILD_REVISION=unknown
ARG RELIQUARY_INSTALL_LOGIC=0

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

# CUDA 13 requires an NVIDIA 580-series or newer host driver.
RUN pip install --upgrade pip wheel setuptools \
 && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

# Build against this Torch ABI; released wheels target older Torch versions.
# PyTorch 2.13 needs C++20; apply upstream FlashAttention's build-only fix.
RUN pip install ninja packaging \
 && curl -fsSL https://files.pythonhosted.org/packages/6c/b1/b77625e3d11260d2bd6a4fc249a4be9a3da97ff848e2222e810c4f0e0211/flash_attn-2.8.3.post2.tar.gz -o /tmp/flash_attn.tar.gz \
 && echo 'd0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b  /tmp/flash_attn.tar.gz' | sha256sum -c - \
 && tar -xzf /tmp/flash_attn.tar.gz -C /tmp \
 && python -c 'from pathlib import Path; p = Path("/tmp/flash_attn-2.8.3.post2/setup.py"); s = p.read_text(); assert s.count("-std=c++17") == 4; p.write_text(s.replace("-std=c++17", "-std=c++20", 2))' \
 && FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS='80;90;100' MAX_JOBS=2 NVCC_THREADS=1 \
    pip install --no-build-isolation --no-deps /tmp/flash_attn-2.8.3.post2 \
 && rm -rf /tmp/flash_attn.tar.gz /tmp/flash_attn-2.8.3.post2

# flash-linear-attention + causal-conv1d: transformers enables Qwen3.5's fast
# GatedDeltaNet path only when BOTH are importable — with either one missing it
# silently falls back to the torch scan for the whole layer (measured 3.77x
# slower forward at 15.7k tokens; the old "fla alone is enough" assumption ran
# production on the fallback for weeks).
RUN pip install flash-linear-attention==0.5.0
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=1 \
    pip install --no-build-isolation --no-deps causal-conv1d==1.6.2.post1

# Source + install
WORKDIR /opt/reliquary
COPY . /opt/reliquary
RUN printf '%s\n' "${RELIQUARY_BUILD_REVISION}" \
      > /opt/reliquary/.build-revision \
 && pip install -e .

# boto3 for R2 (weight-only mode + trainer archive uploads)
RUN pip install boto3

# wandb for trainer telemetry (lazy-imported in reliquary.validator.telemetry).
# No-op at runtime if WANDB_API_KEY is unset. OpenTelemetry is held at the
# Logic runtime lock's 1.44.0, or `pip check` fails when that lock installs.
RUN pip install wandb "opentelemetry-api==1.44.0" "opentelemetry-sdk==1.44.0"

# Explicit candidate variant only. Hash-check before install, then verify the
# installed artifact/import/reward. A local release candidate can be supplied
# with --secret id=logic_wheel,src=/absolute/path/to/the/reviewed.whl.
RUN --mount=type=secret,id=logic_wheel,target=/tmp/logic.whl \
    case "${RELIQUARY_INSTALL_LOGIC}" in \
      0) ;; \
      1) if [ -f /tmp/logic.whl ]; then \
           python scripts/install_logic_runtime.py --wheel /tmp/logic.whl; \
         else python scripts/install_logic_runtime.py; fi ;; \
      *) echo "RELIQUARY_INSTALL_LOGIC must be 0 or 1" >&2; exit 2 ;; \
    esac
LABEL ai.reliquary.logic-runtime="${RELIQUARY_INSTALL_LOGIC}"

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
