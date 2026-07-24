# No-op: the SPEC binaries are run directly under PIN by absolute path (see
# workloads_db binary_cmd), so we do NOT source shrc. shrc is SPEC's runcpu
# harness bootstrap and insists the (bind-mounted, read-only) cpu2017 tree be
# writable, which fails with "not allowed to write into .../config". We don't
# call runcpu and the launch command sets no env from shrc, so nothing needs it.
:
