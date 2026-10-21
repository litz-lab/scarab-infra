#!/bin/bash
# Mirror the traces workloads_db.json refers to into Cloudflare R2, so traces
# come from object storage instead of Google Drive (gdown's cookie prompts and
# rate limits).
#
# The database decides what is uploaded, not a directory walk. The trace store
# also holds superseded trace generations, a second copy of the same simpoints
# at another segment length, top-3 copies of traces that already exist under
# their own suite, raw pipeline output and clustering scratch. A zip nobody can
# reach through workloads_db.json is not a trace we distribute.
#
# Keys mirror the layout both the store and sci already use:
#   <suite>/<subsuite>/<workload>/traces/simp/<cluster_id>.zip
#
# Credentials come from the environment, never the command line (argv is world
# readable through ps) and never a file this script writes:
#   export R2_ACCOUNT_ID=...  R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...
#   export R2_BUCKET=scarabtraces           # default
# An rclone remote named by --remote is used instead when it already exists.
#
#   scripts/sync_traces_r2.sh --dry-run        # plan only
#   scripts/sync_traces_r2.sh                  # upload
#   scripts/sync_traces_r2.sh --verify         # compare both sides
#   scripts/sync_traces_r2.sh --prune          # delete what the db does not list
#
# Re-running is how an interrupted upload finishes: rclone skips what is already
# there, so the script is restartable and safe to put behind a cron entry.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACES_DIR="${TRACES_DIR:-/soe/hlitz/lab/traces}"
R2_BUCKET="${R2_BUCKET:-scarabtraces}"
PREFIX="${R2_PREFIX:-traces}"
REMOTE=""
LOG_DIR="${R2_LOG_DIR:-$HOME/logs}"
TRANSFERS="${R2_TRANSFERS:-16}"
CHECKERS="${R2_CHECKERS:-32}"
BWLIMIT="${R2_BWLIMIT:-}"
# Which suites are held back lives in scripts/trace_manifest.py, so the
# uploader and sci agree without a second list; R2_SKIP overrides it.
read -r -a SKIP <<< "${R2_SKIP:-}"
DRY_RUN=0
VERIFY=0
PRUNE=0

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'
    cat <<'EOF'
Options:
  --dry-run       Show what would transfer; transfer nothing.
  --verify        Compare the manifest against the bucket; upload nothing.
  --prune         Delete bucket objects the database does not list. Asks first.
  --remote NAME   Use an existing rclone remote instead of R2_* env vars.
  --skip "A B"    Suites, or suite/subsuite, to leave out; overrides R2_SKIP.
  --bwlimit RATE  Passed to rclone, e.g. 200M, or "08:00,50M 20:00,off".
  --transfers N   Parallel transfers (default 16).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --verify) VERIFY=1; shift ;;
        --prune) PRUNE=1; shift ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --skip) read -r -a SKIP <<< "$2"; shift 2 ;;
        --bwlimit) BWLIMIT="$2"; shift 2 ;;
        --transfers) TRANSFERS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v rclone >/dev/null || { echo "rclone is not installed" >&2; exit 1; }
[[ -d "$TRACES_DIR" ]] || { echo "no trace directory at $TRACES_DIR" >&2; exit 1; }

if [[ -n "$REMOTE" ]]; then
    rclone listremotes | grep -qx "${REMOTE}:" \
        || { echo "rclone has no remote '${REMOTE}:'" >&2; exit 1; }
    DEST="${REMOTE}:${R2_BUCKET}"
elif [[ -n "${R2_ACCOUNT_ID:-}" && -n "${R2_ACCESS_KEY_ID:-}" && -n "${R2_SECRET_ACCESS_KEY:-}" ]]; then
    export RCLONE_S3_PROVIDER=Cloudflare
    export RCLONE_S3_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    export RCLONE_S3_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    export RCLONE_S3_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    export RCLONE_S3_REGION=auto
    # An API token scoped to one bucket cannot HeadBucket, which rclone would
    # otherwise do before every transfer.
    export RCLONE_S3_NO_CHECK_BUCKET=true
    DEST=":s3:${R2_BUCKET}"
else
    cat >&2 <<EOF
No destination. Either set the R2 credentials:

  export R2_ACCOUNT_ID=<cloudflare account id>
  export R2_ACCESS_KEY_ID=<r2 access key id>
  export R2_SECRET_ACCESS_KEY=<r2 secret>
  export R2_BUCKET=${R2_BUCKET}

or create a remote once and pass it with --remote:

  rclone config create r2 s3 provider=Cloudflare region=auto \\
      endpoint=https://<account id>.r2.cloudflarestorage.com \\
      access_key_id=<key> secret_access_key=<secret> no_check_bucket=true
EOF
    exit 1
fi
[[ -n "$PREFIX" ]] && DEST="${DEST}/${PREFIX}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync_traces_r2_$(date -u +%Y%m%dT%H%M%SZ).log"
MANIFEST="$(mktemp)"
REMOTE_LIST="$(mktemp)"
EXTRAS="$(mktemp)"
trap 'rm -f "$MANIFEST" "$REMOTE_LIST" "$EXTRAS"' EXIT

manifest_args=(--traces-dir "$TRACES_DIR")
[[ ${#SKIP[@]} -gt 0 ]] && manifest_args+=(--skip "${SKIP[@]}")
skipped_note="${SKIP[*]:-$(python3 -c "import sys; sys.path.insert(0, '$REPO_ROOT/scripts'); import trace_manifest; print(' '.join(trace_manifest.DEFAULT_SKIP))")}"
python3 "$REPO_ROOT/scripts/trace_manifest.py" "${manifest_args[@]}" | sort > "$MANIFEST"
absent=$(python3 "$REPO_ROOT/scripts/trace_manifest.py" "${manifest_args[@]}" --missing | wc -l)

echo "trace store : $TRACES_DIR"
echo "destination : $DEST"
echo "manifest    : $(wc -l < "$MANIFEST") files listed by workloads_db.json"
echo "skipping    : ${skipped_note:-nothing}"
[[ $absent -gt 0 ]] && echo "note        : $absent db entries have no file in the store"
echo "log         : $LOG_FILE"
echo

if [[ $VERIFY -eq 1 ]]; then
    rclone check "$TRACES_DIR" "$DEST" --files-from "$MANIFEST" --size-only \
        --log-file "$LOG_FILE" --log-level INFO
    echo "every file the database lists is in the bucket, at the right size."
    exit 0
fi

if [[ $PRUNE -eq 1 ]]; then
    rclone lsf -R --files-only "$DEST" | sort > "$REMOTE_LIST"
    comm -13 "$MANIFEST" "$REMOTE_LIST" > "$EXTRAS"
    count=$(wc -l < "$EXTRAS")
    if [[ $count -eq 0 ]]; then
        echo "nothing in the bucket that the database does not list."
        exit 0
    fi
    echo "$count object(s) in the bucket are not listed by the database:"
    head -10 "$EXTRAS" | sed 's/^/  /'
    [[ $count -gt 10 ]] && echo "  ... and $((count - 10)) more"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "(dry run; nothing deleted)"
        exit 0
    fi
    read -r -p "delete these from $DEST? [y/N] " answer
    [[ "$answer" == "y" ]] || { echo "left alone."; exit 0; }
    rclone delete "$DEST" --files-from "$EXTRAS" --log-file "$LOG_FILE" --log-level INFO
    rclone rmdirs "$DEST" --leave-root --log-file "$LOG_FILE" --log-level INFO || true
    echo "deleted $count object(s)."
    exit 0
fi

RCLONE_ARGS=(
    --files-from "$MANIFEST"
    --transfers "$TRANSFERS"
    --checkers "$CHECKERS"
    # Traces are immutable once written, so size alone decides what is already
    # uploaded; modtime comparison would re-upload after any metadata churn.
    --size-only
    --retries 5
    --low-level-retries 20
    --stats 30s
    --stats-one-line
    --log-file "$LOG_FILE"
    --log-level INFO
)
[[ -n "$BWLIMIT" ]] && RCLONE_ARGS+=(--bwlimit "$BWLIMIT")
[[ $DRY_RUN -eq 1 ]] && RCLONE_ARGS+=(--dry-run)
[[ -t 1 ]] && RCLONE_ARGS+=(--progress)

if rclone copy "$TRACES_DIR" "$DEST" "${RCLONE_ARGS[@]}"; then
    echo "upload complete: $(wc -l < "$MANIFEST") files."
else
    echo "upload failed; re-run to continue (see $LOG_FILE)" >&2
    exit 1
fi
