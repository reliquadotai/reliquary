# Infrastructure source moved

The maintained infrastructure source is the separate private
`reliquadotai/reliquary-infra` repository. Its initial extraction preserves the
tracked subtree from core `0be0cda0c9a73dc3f08e3af2a07dda9407635aa7` with provenance.

Playbooks, host policy, deployment configuration, migration and private
operational evidence belong there. Runtime code, Dockerfiles, dependency locks,
role artifact builders and portable artifact verification remain in this core
repository. The infrastructure repository pins the exact core commit it deploys.

Do not recreate deployment files in this directory or put private handoffs in
the public core repository.
