#!/usr/bin/env python3
"""
Raw AGI Script - No pyst2 library
Direct stdin/stdout communication with Asterisk
"""

import sys
import os

def parse_agi_environment():
    """Parse AGI environment from stdin"""
    env = {}
    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                break
            if ':' in line:
                key, value = line.split(':', 1)
                env[key.strip()] = value.strip()
        except:
            break
    return env

def agi_command(command):
    """Send AGI command to Asterisk"""
    try:
        sys.stdout.write(f"{command}\n")
        sys.stdout.flush()
        result = sys.stdin.readline().strip()
        return result
    except:
        return "ERROR"

def main():
    """Raw AGI main function"""
    try:
        # Parse environment
        env = parse_agi_environment()
        caller_id = env.get('agi_callerid', 'Unknown')

        # Send to stderr for debugging
        print(f"Raw AGI: Call from {caller_id}", file=sys.stderr)

        # Answer call
        agi_command("VERBOSE \"Raw AGI answering call\" 1")
        result = agi_command("ANSWER")
        print(f"Answer result: {result}", file=sys.stderr)

        # Play message
        agi_command("VERBOSE \"Playing demo message\" 1")
        agi_command("STREAM FILE \"demo-thanks\" \"\"")

        # Wait
        agi_command("WAIT 2")

        # Hangup
        agi_command("VERBOSE \"Raw AGI call completed\" 1")
        agi_command("HANGUP")

        print("Raw AGI completed successfully", file=sys.stderr)

    except Exception as e:
        print(f"Raw AGI Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
