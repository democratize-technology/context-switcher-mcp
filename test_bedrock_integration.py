#!/usr/bin/env python3
"""
Test Bedrock integration for Context-Switcher MCP
Verifies the message format fix works correctly
"""

import asyncio
import json
from src.context_switcher_mcp import Thread, ThreadOrchestrator, ModelBackend

async def test_bedrock_format():
    """Test that Bedrock message formatting works"""
    print("Testing Bedrock Message Format")
    print("=" * 40)
    
    # Create a test thread
    thread = Thread(
        id="test-1",
        name="test_perspective",
        system_prompt="You are a helpful assistant. Reply with a brief greeting.",
        model_backend=ModelBackend.BEDROCK,
        model_name="anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    
    # Add a message
    thread.add_message("user", "Hello!")
    
    # Create orchestrator
    orchestrator = ThreadOrchestrator()
    
    try:
        # Test the call
        print("Calling Bedrock with fixed message format...")
        response = await orchestrator._call_bedrock(thread)
        
        if "Error" in response:
            print(f"❌ Bedrock call failed: {response}")
            # Check if it's the old format error
            if "Invalid type for parameter messages" in response:
                print("   Still seeing format error - fix may not be complete")
            else:
                print("   Different error - could be credentials or network")
        else:
            print(f"✅ Bedrock call succeeded!")
            print(f"   Response: {response[:100]}...")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        
    print("\nMessage format being sent:")
    print(json.dumps([{"role": "user", "content": [{"text": "Hello!"}]}], indent=2))

if __name__ == "__main__":
    print("Bedrock Integration Test")
    print("Note: Requires AWS credentials configured")
    print()
    asyncio.run(test_bedrock_format())
