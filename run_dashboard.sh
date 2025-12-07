#!/bin/bash

# Trading Dashboard Launch Script

echo "🚀 Starting Trading Dashboard..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Run Streamlit app
echo "✅ Launching dashboard..."
echo ""
streamlit run trading_dashboard.py

