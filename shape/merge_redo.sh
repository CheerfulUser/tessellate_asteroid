#!/bin/bash
# Merge the doubled-period refits back into the catalog.
#
# The 80 objects in redo_targets.csv had their periods corrected by the doubling pass, so the
# models in shapes/out were fit at half the true rotation and are wrong. The redo array wrote
# replacements to shapes/out_redo -- a separate directory, because the status files are named by
# array task index and an 8-task array would otherwise clobber _status_0000.. from the main
# 100-task run.
#
# Run after array 16577454 completes.
set -euo pipefail

REMOTE=ozstar.swin.edu.au
RBASE=/fred/oz335/rridden/asteroids/shapes
LOCAL=/Users/rridden/Documents/work/code/tess/tessellate_asteroid
PULL=/private/tmp/claude-501/-Users-rridden/52413e60-8be5-4a91-9979-154f632cbcb7/scratchpad/redo_pull

echo "== remote tally"
ssh -o BatchMode=yes "$REMOTE" "ls $RBASE/out_redo/*.json 2>/dev/null | grep -v _status | wc -l" \
  | sed 's/^/  models produced: /'

echo "== promoting refits into out/ (keeps the main run's status files intact)"
ssh -o BatchMode=yes "$REMOTE" "
  cd $RBASE/out_redo
  n=0
  for f in *.json; do
    case \"\$f\" in _status_*) continue;; esac
    cp -f \"\$f\" $RBASE/out/\"\$f\"; n=\$((n+1))
  done
  echo \"  promoted \$n models\"
"

echo "== pulling refits"
rm -rf "$PULL"; mkdir -p "$PULL"
ssh -o BatchMode=yes "$REMOTE" \
  "cd $RBASE/out_redo && tar czf - --exclude='_status_*' *.json" > "$PULL/redo.tar.gz"
tar xzf "$PULL/redo.tar.gz" -C "$PULL" && rm "$PULL/redo.tar.gz"
echo "  pulled: $(ls "$PULL"/*.json 2>/dev/null | wc -l | tr -d ' ')"

echo "== rebuilding those pages"
cd "$LOCAL"
python3 shape/build_pages.py "$PULL"
python3 shape/build_landing.py

echo "== done -- review, then commit and push"
git status --short | head
