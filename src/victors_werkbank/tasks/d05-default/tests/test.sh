#!/bin/sh
set -eu
mkdir -p /var/lib/werkbank-exec /harness /work
chown -R 999:999 /tests /trusted /var/lib/werkbank-exec /harness /work
chmod -R go-rwx /tests /trusted /var/lib/werkbank-exec
chown 999:999 /logs/verifier
chmod 755 /logs/verifier
exec setpriv --reuid=999 --regid=999 --clear-groups \
  --inh-caps=-all --ambient-caps=-all \
  python3 -I /tests/grader.py
