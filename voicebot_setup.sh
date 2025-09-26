#!/bin/bash
# NETOVO VoiceBot Setup Script
# Run this script to set up the production voice bot

set -e

echo "Setting up NETOVO Production Voice Bot..."

# Configuration
VOICEBOT_DIR="/home/aiadmin/netovo_voicebot"
ASTERISK_USER="asterisk"
LOG_DIR="/var/log/asterisk"
SOUNDS_DIR="/var/lib/asterisk/sounds/custom"
MONITOR_DIR="/var/spool/asterisk/monitor"

# Create necessary directories
echo "Creating directories..."
sudo mkdir -p $SOUNDS_DIR
sudo mkdir -p $MONITOR_DIR
sudo mkdir -p "$VOICEBOT_DIR/telephony"

# Set up permissions
echo "Setting up permissions..."
sudo chown -R $ASTERISK_USER:$ASTERISK_USER $SOUNDS_DIR
sudo chown -R $ASTERISK_USER:$ASTERISK_USER $MONITOR_DIR
sudo chmod 755 $SOUNDS_DIR
sudo chmod 755 $MONITOR_DIR

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y sox python3-pip python3-venv

# Set up Python virtual environment
echo "Setting up Python environment..."
cd $VOICEBOT_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Make AGI script executable
echo "Setting up AGI script..."
chmod +x "$VOICEBOT_DIR/telephony/production_agi_voicebot.py"

# Create symbolic link for easier access
sudo ln -sf "$VOICEBOT_DIR/telephony/production_agi_voicebot.py" /usr/local/bin/netovo_voicebot_agi

# Set up logging
echo "Setting up logging..."
sudo touch "$LOG_DIR/voicebot.log"
sudo touch "$LOG_DIR/voicebot_calls.log"
sudo chown $ASTERISK_USER:$ASTERISK_USER "$LOG_DIR/voicebot.log"
sudo chown $ASTERISK_USER:$ASTERISK_USER "$LOG_DIR/voicebot_calls.log"

# Configure logrotate
sudo tee /etc/logrotate.d/voicebot > /dev/null << 'EOF'
/var/log/asterisk/voicebot.log /var/log/asterisk/voicebot_calls.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    copytruncate
    notifempty
    create 644 asterisk asterisk
}
EOF

# Create systemd service for health monitoring
echo "Creating health monitoring service..."
sudo tee /etc/systemd/system/voicebot-health.service > /dev/null << EOF
[Unit]
Description=NETOVO VoiceBot Health Monitor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$VOICEBOT_DIR
ExecStart=$VOICEBOT_DIR/venv/bin/python $VOICEBOT_DIR/health_monitor.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Create health monitor script
cat > "$VOICEBOT_DIR/health_monitor.py" << 'EOF'
#!/usr/bin/env python3
"""
VoiceBot Health Monitor
Monitors component health and restarts services if needed
"""

import time
import subprocess
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ollama_client import OllamaClient
from riva_client import RivaASRClient, RivaTTSClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - HealthMonitor - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/asterisk/voicebot_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_ollama():
    """Check Ollama health"""
    try:
        client = OllamaClient()
        return client.health_check()
    except Exception as e:
        logger.error(f"Ollama check failed: {e}")
        return False

def check_riva():
    """Check RIVA services"""
    try:
        asr = RivaASRClient()
        tts = RivaTTSClient()
        return asr.connect() and tts.connect()
    except Exception as e:
        logger.error(f"RIVA check failed: {e}")
        return False

def restart_riva():
    """Restart RIVA container"""
    try:
        logger.info("Restarting RIVA container...")
        subprocess.run(['sudo', 'docker', 'restart', 'riva-speech'], check=True)
        time.sleep(30)  # Wait for RIVA to start
        return True
    except Exception as e:
        logger.error(f"RIVA restart failed: {e}")
        return False

def main():
    """Main health monitoring loop"""
    logger.info("VoiceBot Health Monitor started")
    
    while True:
        try:
            # Check Ollama
            if not check_ollama():
                logger.warning("Ollama health check failed")
                # You might want to restart Ollama here
            
            # Check RIVA
            if not check_riva():
                logger.warning("RIVA health check failed, attempting restart...")
                restart_riva()
            
            logger.info("Health check completed")
            time.sleep(300)  # Check every 5 minutes
            
        except KeyboardInterrupt:
            logger.info("Health monitor stopped")
            break
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
EOF

chmod +x "$VOICEBOT_DIR/health_monitor.py"

# Enable and start health monitor
sudo systemctl enable voicebot-health.service
sudo systemctl start voicebot-health.service

# Create test script
cat > "$VOICEBOT_DIR/test_voicebot.py" << 'EOF'
#!/usr/bin/env python3
"""
Test script for VoiceBot components
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ollama_client import OllamaClient
from riva_client import RivaASRClient, RivaTTSClient

def test_components():
    """Test all VoiceBot components"""
    print("Testing VoiceBot components...")
    
    # Test Ollama
    print("Testing Ollama...")
    try:
        ollama = OllamaClient()
        if ollama.health_check():
            response = ollama.generate("Hello", max_tokens=10)
            print(f"✓ Ollama working: {response[:50]}")
        else:
            print("✗ Ollama health check failed")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
    
    # Test RIVA
    print("Testing RIVA...")
    try:
        asr = RivaASRClient()
        tts = RivaTTSClient()
        
        if asr.connect() and tts.connect():
            print("✓ RIVA services connected")
            
            # Test TTS
            tts_file = tts.synthesize("Hello from NETOVO voice bot")
            if tts_file:
                print(f"✓ TTS working: {tts_file}")
                
                # Test ASR
                transcript = asr.transcribe_file(tts_file)
                print(f"✓ ASR working: {transcript}")
            else:
                print("✗ TTS failed")
        else:
            print("✗ RIVA connection failed")
    except Exception as e:
        print(f"✗ RIVA error: {e}")

if __name__ == "__main__":
    test_components()
EOF

chmod +x "$VOICEBOT_DIR/test_voicebot.py"

# Update Asterisk configuration
echo "Updating Asterisk configuration..."
sudo cp /etc/asterisk/extensions.conf /etc/asterisk/extensions.conf.backup

# Reload Asterisk configuration
echo "Reloading Asterisk..."
sudo asterisk -rx "core reload"
sudo asterisk -rx "dialplan reload"

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Test components: $VOICEBOT_DIR/test_voicebot.py"
echo "2. Check logs: tail -f /var/log/asterisk/voicebot.log"
echo "3. Make a test call to extension 1600"
echo ""
echo "Monitoring:"
echo "- Health monitor status: sudo systemctl status voicebot-health"
echo "- Call logs: tail -f /var/log/asterisk/voicebot_calls.log"
echo ""
