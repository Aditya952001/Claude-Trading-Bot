#!/usr/bin/env bash
# ==============================================================================
# setup_oracle.sh - Automated Oracle Cloud (Always Free VM) One-Click Setup
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "    Deriv Autonomous Bot - Oracle Cloud Setup Assistant    "
echo "============================================================"

# 1. Install System Dependencies
echo "[1/4] Installing Python3, venv, and system packages..."
sudo apt update -y
sudo apt install -y python3-venv python3-pip rsync sqlite3

# 2. Virtual Environment
echo "[2/4] Setting up Python virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create Logs & Data directories
echo "[3/4] Preparing data and logs directories..."
mkdir -p logs data

# 4. Systemd 24/7 Background Service Setup
echo "[4/4] Configuring 24/7 background systemd service..."
USER_NAME="$(whoami)"
SERVICE_FILE="/etc/systemd/system/deriv-bot.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=Deriv Rise/Fall High-Conviction Autonomous Trading Bot
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/.venv/bin/python run_rise_fall.py
Restart=always
RestartSec=10
EnvironmentFile=$SCRIPT_DIR/.env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable deriv-bot

echo "============================================================"
echo " Setup complete!"
echo ""
echo " Useful Commands:"
echo "   Start the bot:    sudo systemctl start deriv-bot"
echo "   Stop the bot:     sudo systemctl stop deriv-bot"
echo "   Restart the bot:  sudo systemctl restart deriv-bot"
echo "   View live logs:   tail -f logs/bot.log"
echo "   Service status:   sudo systemctl status deriv-bot"
echo "============================================================"
