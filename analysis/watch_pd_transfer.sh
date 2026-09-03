#!/bin/bash
# Poll the PD transfer cells and regenerate the transferability figures each
# time a new one lands.  Exits when all 18 PD cells are in, or after MAX_HOURS.
#
#   nohup analysis/watch_pd_transfer.sh > logs/watch_pd_transfer.log 2>&1 &
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

MAX_HOURS=${MAX_HOURS:-12}
INTERVAL=${INTERVAL:-600}
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
last=""

while :; do
    ready=$(python analysis/harvest_pd_transfer.py | tee /dev/stderr | grep -oP '^\d+(?=/18 PD cells ready)')
    if [[ "$ready" != "$last" ]]; then
        echo "[watch] $(date -Is)  $ready/18 ready -> regenerating figures"
        python analysis/plot_transferability.py --table all
        last="$ready"
    fi
    [[ "$ready" == "18" ]] && { echo "[watch] all PD cells in"; break; }
    [[ $(date +%s) -ge $deadline ]] && { echo "[watch] deadline reached at $ready/18"; break; }
    sleep "$INTERVAL"
done

echo "[watch] final state:"
python analysis/harvest_pd_transfer.py --print | tail -20
squeue -u "$USER" -h -o "%j %T" | grep -c "transfer.*pd\|pd.*transfer" || true
