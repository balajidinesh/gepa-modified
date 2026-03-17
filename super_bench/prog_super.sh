#!/bin/bash

echo "🚀 Script started with PID: $$"
echo "You can kill it using: kill $$"
echo $$ > benchmark_runner_rcb.pid


PID=35032


echo "⏳ Waiting for current training to finish (PID: $PID)..."
tail --pid=$PID -f /dev/null


echo "🧹 Cleaning Docker..."

docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

echo "✅ Docker cleaned."


echo "✅ prev finished. Starting next program..."
python super_bench/prog_gepa-iters-super_1.py




echo "🧹 Cleaning Docker..."

docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

echo "✅ Docker cleaned."


echo "✅ prev finished. Starting next program..."
python super_bench/prog_gepa-iters-super_2
.py