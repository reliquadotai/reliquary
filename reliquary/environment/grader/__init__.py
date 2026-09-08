"""Grader subprocess components for the code-execution environment."""

from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE

# Shared defaults must not import the controller's environment/corpus registry.
GRADER_SOCKET_PATH = "/tmp/reliquary-grader.sock"
GRADER_POOL_SIZE = 4 * ACTIVE_PROTOCOL_PROFILE.sampling.rollouts
GRADER_EVAL_TIMEOUT_SECONDS = 5
