#!/bin/bash
# Waits for the phase-1 queue jobs to finish their current runs (they exit because
# everything else is HOLD-ed), then submits FD002 queue jobs in their place.
# Needed because the mtech QOS rejects a 3rd submitted job. Run detached on the login node:
#   setsid nohup bash slurm/resubmit_when_free.sh > slurm/logs/watcher.log 2>&1 &
DK=/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman
cd $DK || exit 1
TASKS=tasks/fd002.txt
WANT=2
submitted=0
released=0
while [ $submitted -lt $WANT ]; do
    njobs=$(squeue -u $USER -h | wc -l)
    if [ "$njobs" -lt 2 ]; then
        if [ $released -eq 0 ]; then
            # un-park the FD002 tasks of the new list that phase 1 had HOLD-ed
            grep -v '^#' $TASKS | while read -r line; do
                set -- $line
                id="${8}_${2}_${4}_${6}_run${10}"
                [ -f runs/.claims/$id/HOLD ] && rm -rf runs/.claims/$id && echo "$(date +%T) released $id"
            done
            released=1
        fi
        out=$(sbatch -J dk_fd002_$((submitted + 1)) slurm/queue.slurm $TASKS 2>&1)
        echo "$(date '+%F %T') $out"
        case "$out" in Submitted*) submitted=$((submitted + 1)) ;; esac
    fi
    sleep 60
done
echo "$(date '+%F %T') watcher done"
