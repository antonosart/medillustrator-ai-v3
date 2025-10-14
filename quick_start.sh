#!/bin/bash
set -e

PORT=8501
APP_FILE="app_v3_langgraph.py"

echo "🚀 Starting MedIllustrator-AI v3.0..."

# Check if app file exists
if [[ ! -f "$APP_FILE" ]]; then
    echo "❌ Error: $APP_FILE not found!"
    exit 1
fi

# Activate venv if exists
if [[ -d "venv" ]]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
fi

# Create required directories
mkdir -p logs cache data uploads

# Kill existing process if any
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠ Stopping existing application..."
    kill -9 $(lsof -ti:$PORT) 2>/dev/null || true
    sleep 2
fi

echo ""
echo "✓ Starting application on port $PORT..."
echo "📱 Open browser: http://localhost:$PORT"
echo "🛑 Press Ctrl+C to stop"
echo ""

streamlit run "$APP_FILE" --server.port=$PORT --server.address=0.0.0.0
