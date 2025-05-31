#!/bin/bash

# AutoGen Multi-Agent Personal Assistant Installation Script

set -e  # Exit on error

echo "🤖 AutoGen Multi-Agent Personal Assistant Setup"
echo "================================================"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "❌ Error: Python 3.9+ is required. Current version: $PYTHON_VERSION"
    echo "Please install Python 3.9 or higher and try again."
    exit 1
fi

echo "✅ Python version check passed: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed from requirements.txt"
elif [ -f "pyproject.toml" ]; then
    pip install -e .
    echo "✅ Dependencies installed from pyproject.toml"
else
    echo "❌ Error: No requirements.txt or pyproject.toml found"
    exit 1
fi

# Check if model_config.yml exists
echo ""
echo "⚙️  Checking configuration..."
if [ ! -f "model_config.yml" ]; then
    if [ -f "model_config_template.yml" ]; then
        cp model_config_template.yml model_config.yml
        echo "⚠️  Created model_config.yml from template"
        echo "🔧 Please edit model_config.yml with your API keys and settings"
    else
        echo "❌ Error: model_config_template.yml not found"
        exit 1
    fi
else
    echo "✅ model_config.yml exists"
fi

# Create API credentials files if they don't exist
echo ""
echo "🔑 Setting up API credentials..."

# Google Maps API
if [ ! -f "google_maps_api_key.txt" ]; then
    echo "📍 Creating placeholder for Google Maps API key..."
    echo "# Replace with your Google Maps API key" > google_maps_api_key.txt
    echo "YOUR_GOOGLE_MAPS_API_KEY_HERE" >> google_maps_api_key.txt
    echo "⚠️  Please add your Google Maps API key to google_maps_api_key.txt"
fi

# Google Calendar credentials
if [ ! -f "credentials.json" ]; then
    echo "📅 Creating placeholder for Google Calendar credentials..."
    cat > credentials.json << 'EOF'
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
EOF
    echo "⚠️  Please replace credentials.json with your Google Calendar API credentials"
fi

# Make scripts executable
chmod +x run.sh 2>/dev/null || true

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Edit model_config.yml with your OpenAI/Azure API credentials"
echo "2. Add your Google Maps API key to google_maps_api_key.txt"
echo "3. Replace credentials.json with your Google Calendar API credentials"
echo "4. Run the assistant with: ./run.sh or python3 orchestrator.py"
echo ""
echo "💡 To activate the virtual environment manually:"
echo "   source venv/bin/activate"
echo ""
echo "📚 For more information, see README.md" 