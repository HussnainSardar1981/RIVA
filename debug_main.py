#!/usr/bin/env python3
"""
Debug script to identify the exact issue
"""
import asyncio
import sys
import os

def debug_imports():
    """Test imports one by one"""
    print("=== Testing Imports ===")
    
    try:
        from ollama_client import OllamaClient, VOICE_BOT_SYSTEM_PROMPT
        print("✅ ollama_client imported successfully")
        
        # Test the client creation
        client = OllamaClient()
        print("✅ OllamaClient created successfully")
        
        # Test health check
        health = client.health_check()
        print(f"Health check result: {health}")
        
        return client
        
    except Exception as e:
        print(f"❌ Import/creation failed: {e}")
        sys.exit(1)

async def debug_generation(client):
    """Test generation"""
    print("\n=== Testing Generation ===")
    
    try:
        response = await client.generate("Hello", max_tokens=5)
        print(f"✅ Generation successful: '{response}'")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

async def debug_main_components():
    """Test main.py components without full initialization"""
    print("\n=== Testing Main Components ===")
    
    try:
        # Import the VoiceBot class
        from main import VoiceBot
        print("✅ VoiceBot imported")
        
        # Create instance
        bot = VoiceBot()
        print("✅ VoiceBot created")
        
        # Test just the Ollama component
        health = bot.ollama.health_check()
        print(f"VoiceBot Ollama health: {health}")
        
        if health:
            response = await bot.ollama.generate("Test", max_tokens=3)
            print(f"✅ VoiceBot generation: '{response}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Main component test failed: {e}")
        return False

async def main():
    print("🔍 Debugging Voice Bot Issues\n")
    
    # Step 1: Test imports and basic functionality
    client = debug_imports()
    
    # Step 2: Test generation
    gen_ok = await debug_generation(client)
    
    # Step 3: Test main.py components
    main_ok = await debug_main_components()
    
    # Summary
    print("\n=== Debug Summary ===")
    if gen_ok and main_ok:
        print("✅ All components working - main.py should work now")
    else:
        print("❌ Issues found - see errors above")
        
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
