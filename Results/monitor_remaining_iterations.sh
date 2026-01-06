#!/bin/bash
# Auto-monitoring script for remaining Bayesian Optimization iterations
# Created: 2026-01-05 21:45

LOG="/home/developer/Desktop/nt_project/MatlabProject/matlab_training.log"
REPORT="/home/developer/Desktop/nt_project/MatlabProject/Results/BAYESIAN_OPT_FINAL_REPORT.txt"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    BAYESIAN OPTIMIZATION - AUTO MONITOR (Iter 7-10)          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Monitoring log file: $LOG"
echo "Final report will be saved to: $REPORT"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

LAST_ITER=7

while true; do
    # Check if iteration 7 completed
    if grep -q "Random Search Iteration 8/10" "$LOG" 2>/dev/null; then
        if [ $LAST_ITER -lt 8 ]; then
            echo "[$(date '+%H:%M:%S')] ✅ Iteration 7 COMPLETED"
            LAST_ITER=8
        fi
    fi

    # Check if iteration 8 completed
    if grep -q "Random Search Iteration 9/10" "$LOG" 2>/dev/null; then
        if [ $LAST_ITER -lt 9 ]; then
            echo "[$(date '+%H:%M:%S')] ✅ Iteration 8 COMPLETED"
            LAST_ITER=9
        fi
    fi

    # Check if iteration 9 completed
    if grep -q "Random Search Iteration 10/10" "$LOG" 2>/dev/null; then
        if [ $LAST_ITER -lt 10 ]; then
            echo "[$(date '+%H:%M:%S')] ✅ Iteration 9 COMPLETED"
            LAST_ITER=10
        fi
    fi

    # Check if optimization completed
    if grep -q "OPTIMIZATION COMPLETE" "$LOG" 2>/dev/null; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[$(date '+%H:%M:%S')] 🎉 ALL 10 ITERATIONS COMPLETED!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Generate final report
        echo "Generating final report..."
        python3 /home/developer/Desktop/nt_project/MatlabProject/Results/generate_final_report.py

        echo ""
        echo "✅ Final report saved to: $REPORT"
        echo ""
        echo "📋 Next step: Check the report and manually select best hyperparameters"
        echo "   (Kod otomatik MSE'ye göre seçecek ama RMSE'ye göre seçmelisin!)"
        break
    fi

    # Show current iteration progress every 60 seconds
    CURRENT_ITER=$(tail -100 "$LOG" | grep "Random Search Iteration" | tail -1 | grep -oP '\d+(?=/10)')
    if [ ! -z "$CURRENT_ITER" ]; then
        LAST_LINE=$(tail -1 "$LOG")
        if [[ $LAST_LINE == *"|"* ]]; then
            EPOCH=$(echo "$LAST_LINE" | awk '{print $2}')
            ITER_NUM=$(echo "$LAST_LINE" | awk '{print $4}')
            TRAIN_RMSE=$(echo "$LAST_LINE" | awk '{print $6}')
            VAL_RMSE=$(echo "$LAST_LINE" | awk '{print $8}')
            echo "[$(date '+%H:%M:%S')] Iter $CURRENT_ITER - Epoch $EPOCH, Iteration $ITER_NUM, Val RMSE: $VAL_RMSE"
        fi
    fi

    sleep 60
done
