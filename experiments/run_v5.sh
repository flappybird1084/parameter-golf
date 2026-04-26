#!/bin/bash
cd /root/parameter-golf
LOG="experiments/v5_run1.log"
echo "Starting v5 (XSA-all, 9L, standard eval) at $(date)"
python3 experiments/v5_xsa.py >& $LOG &
PID=$!
echo "PID: $PID"
echo "Log: $LOG"
# Monitor
tail -f $LOG &
TAIL_PID=$!
wait $PID
kill $TAIL_PID 2>/dev/null
echo "v5 completed at $(date)"
