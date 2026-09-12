#!/bin/bash
# fetch_arxiv.sh -- download the arXiv PDFs of all entries of a BibTeX file (INSPIRE export).
#
#   ./fetch_arxiv.sh                 # all entries of INSPIRE-KamLAND.bib into this directory
#   ./fetch_arxiv.sh -n              # dry run: list what would be downloaded
#   ./fetch_arxiv.sh -f other.bib -d pdfs -s 5
#
# Each entry's "eprint" field (e.g. 2406.11438 or hep-ex/0212021) is fetched from
# https://arxiv.org/pdf/<eprint> and saved as <bibkey>_<eprint>.pdf; existing files are
# skipped, so the script can be re-run.  arXiv asks automated clients to stay well below
# one request per second, hence the pause between downloads (default 3 s).
set -u
cd "$(dirname "$0")"
BIB="INSPIRE-KamLAND.bib"; DEST="."; SLEEP=3; DRY=0
while getopts "f:d:s:nh" opt; do
  case $opt in
    f) BIB=$OPTARG ;;
    d) DEST=$OPTARG ;;
    s) SLEEP=$OPTARG ;;
    n) DRY=1 ;;
    h|*) sed -n 2,12p "$0"; exit 0 ;;
  esac
done
[ -f "$BIB" ] || { echo "no such file: $BIB" >&2; exit 1; }
mkdir -p "$DEST"

# key <tab> eprint for every entry that has an eprint field
entries=$(awk '
  /^@/          { key = $0; sub(/^@[A-Za-z]+[{(] */, "", key); sub(/ *,.*$/, "", key); ep = "" }
  /^[ \t]*eprint[ \t]*=/ { ep = $0; sub(/^[^=]*=[ \t]*["{]*/, "", ep); sub(/["}],?[ \t]*$/, "", ep) }
  /^}/          { if (ep != "") print key "\t" ep }
' "$BIB")

n_ok=0; n_skip=0; n_fail=0; n_total=$(printf "%s\n" "$entries" | grep -c .)
echo "$n_total entries with an arXiv identifier in $BIB"
while IFS=$'\t' read -r key ep; do
  [ -z "$key" ] && continue
  safe_key=${key//:/_}; out="$DEST/${safe_key}_${ep//\//_}.pdf"
  url="https://arxiv.org/pdf/$ep"
  if [ -s "$out" ]; then n_skip=$((n_skip+1)); continue; fi
  if [ $DRY -eq 1 ]; then echo "would fetch $url -> $out"; continue; fi
  printf "%-40s %-18s " "$key" "$ep"
  if curl -sSL --fail --retry 3 --retry-delay 5 -A "fetch_arxiv.sh (mailto: see repository)" -o "$out.part" "$url" \
     && head -c 4 "$out.part" | grep -q "%PDF"; then
    mv "$out.part" "$out"; echo "ok  $(du -k "$out" | cut -f1) kB"; n_ok=$((n_ok+1))
  else
    rm -f "$out.part"; echo "FAILED"; n_fail=$((n_fail+1))
  fi
  sleep "$SLEEP"
done <<< "$entries"
[ $DRY -eq 1 ] || echo "downloaded $n_ok, already present $n_skip, failed $n_fail"
