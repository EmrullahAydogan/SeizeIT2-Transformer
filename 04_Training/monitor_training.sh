#!/bin/bash
# monitor_training.sh - Real-time training monitor
# Usage: ./monitor_training.sh [log_file]

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         TRAINING MONITOR - Real-time Progress Tracker        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Find the most recent training log
RESULTS_DIR="../Results"

if [ -z "$1" ]; then
    LOG_FILE=$(ls -t $RESULTS_DIR/training_log_*.txt 2>/dev/null | head -1)

    if [ -z "$LOG_FILE" ]; then
        echo -e "${RED}No training log found!${NC}"
        echo "Start training first, or specify log file:"
        echo "  ./monitor_training.sh /path/to/training_log.txt"
        exit 1
    fi
else
    LOG_FILE="$1"
fi

METRICS_FILE="${LOG_FILE/training_log/training_metrics}"
METRICS_FILE="${METRICS_FILE/.txt/.csv}"

echo -e "${GREEN}✓ Monitoring log:${NC} $LOG_FILE"
echo -e "${GREEN}✓ Metrics file:${NC} $METRICS_FILE"
echo ""

# Check if files exist
if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}Error: Log file not found: $LOG_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Monitor the log file
tail -f "$LOG_FILE" | while read line; do
    # Highlight important messages
    if [[ $line == *"Epoch"* ]] && [[ $line == *"Train Loss"* ]]; then
        echo -e "${BLUE}$line${NC}"
    elif [[ $line == *"✓ Improvement"* ]]; then
        echo -e "${GREEN}$line${NC}"
    elif [[ $line == *"⚠ No improvement"* ]]; then
        echo -e "${YELLOW}$line${NC}"
    elif [[ $line == *"STOPPING"* ]]; then
        echo -e "${RED}$line${NC}"
    elif [[ $line == *"COMPLETE"* ]] || [[ $line == *"OPTIMIZATION"* ]]; then
        echo -e "${GREEN}$line${NC}"
    else
        echo "$line"
    fi
done
