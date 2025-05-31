#!/bin/bash

# AutoGen Multi-Agent Personal Assistant Run Script

set -e  # Exit on error

echo "🤖 Starting AutoGen Multi-Agent Personal Assistant..."
echo "===================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if configuration exists
if [ ! -f "model_config.yml" ]; then
    echo "❌ model_config.yml not found. Please run ./install.sh first."
    exit 1
fi

# Check if the configuration has been updated from template
if grep -q "REPLACE_WITH_YOUR_API_KEY" model_config.yml; then
    echo "⚠️  Warning: model_config.yml still contains placeholder values."
    echo "Please edit model_config.yml with your actual API credentials."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Please configure your API keys and try again."
        exit 1
    fi
fi

# Start the orchestrator
echo ""
echo "🚀 Launching orchestrator..."
echo ""

python3 orchestrator.py 