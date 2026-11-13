# syntax=docker/dockerfile:1.6
# The SciCode grading service's sandbox root (gVisor, no network): the grader
# rootfs's Python, SciCode's test dependencies, upstream's `scicode` test helpers
# (scicode-bench/SciCode @ e3158ea0, Apache-2.0; `datasets` import made lazy) and
# the reference targets. Built and stored by digest in our registry, then exported
# as a bundle rootfs on the eval host (docs/runbooks/order-control.md).
#
#   docker build -t scicode-grader --build-context data=<dir holding test_data.h5> docker/scicode-grader
ARG PY_IMAGE=python:3.12-slim@sha256:090ba77e2958f6af52a5341f788b50b032dd4ca28377d2893dcf1ecbdfdfe203
FROM ${PY_IMAGE}

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock /tmp/requirements.lock
RUN pip install --require-hashes --no-deps -r /tmp/requirements.lock && rm /tmp/requirements.lock
COPY scicode /usr/local/lib/python3.12/site-packages/scicode
RUN python -m compileall -q /usr/local/lib/python3.12/site-packages/scicode \
 && python -c "import numpy, scipy, sympy, h5py, matplotlib, scicode.parse.parse, scicode.compare.cmp"

# test_data.h5 from the SciCode README's Google Drive folder (file id
# 17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW, 1,049,345,865 bytes). Upstream publishes no
# checksum: this is the file's own sha256.
ARG H5_SHA256=48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890
# The directory first: COPY --chmod would give the one it creates 0444 (not traversable).
RUN mkdir -m 0755 /opt/scicode
COPY --chmod=0444 --from=data test_data.h5 /opt/scicode/test_data.h5
RUN echo "${H5_SHA256}  /opt/scicode/test_data.h5" | sha256sum -c -
