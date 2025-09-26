#!/bin/bash
# Deploy diagnostic AGI script

echo "Deploying diagnostic VoiceBot script..."

# Create the diagnostic script
DIAGNOSTIC_SCRIPT="/home/aiadmin/netovo_voicebot/telephony/diagnostic_agi_voicebot.py"

# Make it executable
chmod +x "$DIAGNOSTIC_SCRIPT"

# Update extensions.conf to use diagnostic script
echo "Updating extensions.conf for diagnostic mode..."

# Backup current extensions.conf
sudo cp /etc/asterisk/extensions.conf /etc/asterisk/extensions.conf.backup

# Create new extensions.conf with diagnostic script
sudo tee /etc/asterisk/extensions.conf > /dev/null << 'EOF'
[general]
static=yes
writeprotect=no

[globals]

[voicebot-incoming]
; Diagnostic VoiceBot context
exten => s,1,NoOp(DIAGNOSTIC Voice Bot - Call from ${CALLERID(num)})
 same => n,Set(CHANNEL(language)=en)
 same => n,Answer()
 same => n,Wait(1)
 same => n,AGI(/home/aiadmin/netovo_voicebot/telephony/diagnostic_agi_voicebot.py)
 same => n,NoOp(Diagnostic AGI completed with status: ${AGISTATUS})
 same => n,Wait(2)
 same => n,Hangup()

; Handle extension 1600 directly
exten => 1600,1,NoOp(Direct call to diagnostic VoiceBot)
 same => n,Goto(s,1)

; Catch all other extensions
exten => _X.,1,NoOp(Catch-all for diagnostic VoiceBot)
 same => n,Goto(s,1)

[default]
; Local test extension
exten => 1600,1,NoOp(Local test call)
 same => n,Goto(voicebot-incoming,s,1)

; Echo test
exten => *777,1,NoOp(Echo Test)
 same => n,Answer()
 same => n,Echo()
EOF

# Reload Asterisk
echo "Reloading Asterisk configuration..."
sudo asterisk -rx "dialplan reload"

# Clear any existing debug logs
sudo rm -f /tmp/voicebot_debug.log

# Set permissions
sudo chmod 666 /tmp/voicebot_debug.log 2>/dev/null || true

echo ""
echo "Diagnostic script deployed!"
echo ""
echo "Next steps:"
echo "1. Make a test call to extension 1600"
echo "2. While call is active, check debug log:"
echo "   tail -f /tmp/voicebot_debug.log"
echo ""
echo "3. After call ends, review full debug log:"
echo "   cat /tmp/voicebot_debug.log"
echo ""
echo "4. Also check Asterisk logs:"
echo "   sudo tail -f /var/log/asterisk/full"
echo ""
