#!/bin/bash

echo "🚀 Script started with PID: $$"
echo "You can kill it using: kill $$"
echo $$ > benchmark_runner_rcb.pid


PID=61306


echo "⏳ Waiting for current training to finish (PID: $PID)..."
tail --pid=$PID -f /dev/null


echo "🧹 Cleaning Docker..."

docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

echo "✅ Docker cleaned."


echo "✅ prev finished. Starting next program..."
python scienceagentbench/variance_sab_wog.py

