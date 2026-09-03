#!/bin/bash
# Download NightbeatDB (40 participants, ~4.5GB) from ETH polybox public share.
# Share: https://polybox.ethz.ch/index.php/s/oszYjY5aZ2xwMZp
# Files are served over Nextcloud public WebDAV; we fetch them individually so
# the download is resumable and verifiable (a single 4.5GB zip is not).
set -uo pipefail

TOKEN="oszYjY5aZ2xwMZp"
DAV="https://polybox.ethz.ch/public.php/dav/files/${TOKEN}"
DEST="/projects/b53017/vbc1591/Nightbeat/raw/nightbeatdb"
MANIFEST="/projects/b53017/vbc1591/Nightbeat/manifest.tsv"
JOBS=6

mkdir -p "$DEST"

# 1. List the share and record name<TAB>size for verification.
if [ ! -s "$MANIFEST" ]; then
  echo "[$(date +%T)] listing share ..."
  curl -s -X PROPFIND -u "${TOKEN}:" -H "Depth: 1" "${DAV}/" \
    | python3 -c '
import sys, re, urllib.parse
t = sys.stdin.read()
# each <d:response> holds one href and (for files) one getcontentlength
for resp in re.findall(r"<d:response>(.*?)</d:response>", t, re.S):
    href = re.search(r"<d:href>(.*?)</d:href>", resp)
    size = re.search(r"<d:getcontentlength>(\d+)</d:getcontentlength>", resp)
    if not href or not size:
        continue
    name = urllib.parse.unquote(href.group(1)).rstrip("/").split("/")[-1]
    if name.endswith(".npz"):
        print(f"{name}\t{size.group(1)}")
' > "$MANIFEST"
fi
N=$(wc -l < "$MANIFEST")
echo "[$(date +%T)] manifest: $N files"

# 2. Fetch anything missing or wrong-sized, JOBS at a time.
fetch_one() {
  local name="$1" want="$2"
  local out="${DEST}/${name}"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then
    return 0
  fi
  for attempt in 1 2 3; do
    curl -sS -f -L --retry 3 --retry-delay 5 --connect-timeout 30 \
         -u "${TOKEN}:" -o "${out}.part" "${DAV}/${name}" && \
    [ "$(stat -c%s "${out}.part")" = "$want" ] && \
    mv "${out}.part" "$out" && echo "[ok] $name" && return 0
    echo "[retry $attempt] $name"
    sleep 10
  done
  rm -f "${out}.part"
  echo "[FAIL] $name"
  return 1
}
export -f fetch_one
export DAV DEST TOKEN

echo "[$(date +%T)] downloading with ${JOBS} workers ..."
awk -F'\t' '{print $1"\t"$2}' "$MANIFEST" \
  | xargs -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r n s <<< "{}"; fetch_one "$n" "$s"'

# 3. Verify.
echo "[$(date +%T)] verifying ..."
BAD=0
while IFS=$'\t' read -r name want; do
  f="${DEST}/${name}"
  if [ ! -f "$f" ] || [ "$(stat -c%s "$f")" != "$want" ]; then
    echo "MISSING/BAD: $name"
    BAD=$((BAD+1))
  fi
done < "$MANIFEST"

if [ "$BAD" -eq 0 ]; then
  echo "[$(date +%T)] DOWNLOAD COMPLETE: $N/$N files, $(du -sh "$DEST" | cut -f1)"
else
  echo "[$(date +%T)] DOWNLOAD INCOMPLETE: $BAD file(s) missing/bad -- rerun this script"
  exit 1
fi
