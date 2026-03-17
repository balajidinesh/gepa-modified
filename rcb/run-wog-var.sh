PID=88570

echo "⏳ Waiting for current training to finish (PID: $PID)..."
tail --pid=$PID -f /dev/null



echo "🧹 Cleaning Docker..."

docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

echo "✅ Docker cleaned."



echo "✅ Training finished. Starting next program..."
python rcb/variance-iters-rcb-wog.py