# syntax=docker/dockerfile:1.6
ARG GRADER_PY_IMAGE=python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203
FROM ${GRADER_PY_IMAGE} AS grader-rootfs
COPY scripts/bootstrap_python_trust.py /opt/bootstrap_python_trust.py
RUN python /opt/bootstrap_python_trust.py --system

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ARG RELIQUARY_BUILD_REVISION=unknown
ARG RELIQUARY_INSTALL_LOGIC=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RELIQUARY_BUILD_REVISION=${RELIQUARY_BUILD_REVISION}

RUN apt-get update -qq && apt-get install -y -qq \
        python3.12 python3.12-venv \
        git build-essential wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv so system pip stays clean
RUN python3.12 -m venv /opt/reliquary-venv
ENV PATH="/opt/reliquary-venv/bin:${PATH}"
COPY scripts/bootstrap_python_trust.py /opt/bootstrap_python_trust.py
RUN python /opt/bootstrap_python_trust.py --system

# torch 2.7.0 + CUDA 12.8 (matches our Targon setup)
RUN pip install --upgrade wheel setuptools \
 && pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# flash-attn prebuilt wheel for torch 2.7 / cu12 / cp312 / cxx11abi=TRUE.
# Do NOT rename the wheel on download: pip parses the version and platform
# tags from the filename and rejects anything that doesn't match the
# ``name-version-...-abi.whl`` shape with "Invalid wheel filename
# (wrong number of parts)".
ARG FA_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
ARG FA_SHA256=cafd4f708725e1be95495a86cbf418cfada08e707c8a797def61488a2a3c4f72
RUN wget -q "${FA_URL}" -P /tmp/ \
 && printf '%s  %s\n' "${FA_SHA256}" /tmp/flash_attn-*.whl | sha256sum -c - \
 && pip install /tmp/flash_attn-*.whl \
 && rm /tmp/flash_attn-*.whl

# flash-linear-attention + causal-conv1d: transformers enables Qwen3.5's fast
# GatedDeltaNet path only when BOTH are importable — with either one missing it
# silently falls back to the torch scan for the whole layer (measured 3.77x
# slower forward at 15.7k tokens; the old "fla alone is enough" assumption ran
# production on the fallback for weeks). Prebuilt wheel: no nvcc needed.
RUN pip install flash-linear-attention==0.5.0
ARG CC1D_URL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
ARG CC1D_SHA256=9f5426f16f8838afc7cc6a914471321f592444efbc7b2584d51a10c0c49bb71a
RUN wget -q "${CC1D_URL}" -P /tmp/ \
 && printf '%s  %s\n' "${CC1D_SHA256}" /tmp/causal_conv1d-*.whl | sha256sum -c - \
 && pip install /tmp/causal_conv1d-*.whl \
 && rm /tmp/causal_conv1d-*.whl

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
RUN python /opt/bootstrap_python_trust.py --system

# ────────────────  GRADER SANDBOX  ────────────────
# Install gVisor (runsc) for the OpenCodeInstruct env's sandbox.
# Pinned to an exact release + verified checksum: the grader's pass/total
# feeds rewards, so a different runsc across validators would break the
# cross-box-determinism guarantee (and ``latest`` changed every rebuild).
# Bump RUNSC_RELEASE deliberately.
ARG RUNSC_RELEASE=20260525
ARG RUNSC_SHA512=7fd4103d6df5202545011c39449cc484371384059db17b917d811c03e8355c829109e9848603b77f8ec1b050d7a1179ed239546059e210edc67601bfa0504bbb
RUN ARCH="$(uname -m)" \
 && test "${ARCH}" = x86_64 \
 && BASE="https://storage.googleapis.com/gvisor/releases/release/${RUNSC_RELEASE}/${ARCH}" \
 && wget -q "${BASE}/runsc" \
 && printf '%s  %s\n' "${RUNSC_SHA512}" runsc | sha512sum -c - \
 && chmod +x runsc \
 && mv runsc /usr/local/bin/runsc

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
