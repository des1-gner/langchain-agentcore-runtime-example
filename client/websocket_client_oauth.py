from bedrock_agentcore.runtime import AgentCoreRuntimeClient
import websockets
import asyncio
import json
import os

async def main():
    """Test WebSocket connection using OAuth authentication"""
    # Get runtime ARN from environment variable
    runtime_arn = os.getenv('AGENT_ARN')
    if not runtime_arn:
        raise ValueError("AGENT_ARN environment variable is required")
    
    # Get OAuth bearer token from environment variable
    bearer_token = os.getenv('BEARER_TOKEN')
    if not bearer_token:
        raise ValueError("BEARER_TOKEN environment variable required for OAuth")
    
    # Initialize client
    client = AgentCoreRuntimeClient(region="us-west-2")
    
    # Generate WebSocket connection with OAuth
    ws_url, headers = client.generate_ws_connection_oauth(
        runtime_arn=runtime_arn,
        bearer_token=bearer_token
    )
    
    test_prompts = [
        "What is the current timestamp?",
        "Generate a random number between 1 and 1000",
        "What is the SHA256 hash of the word 'hello'?",
    ]
    
    try:
        async with websockets.connect(ws_url, additional_headers=headers) as ws:
            print("="*80)
            print("Testing WebSocket Agent with OAuth")
            print("="*80)
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\nTest {i}/{len(test_prompts)}: {prompt}")
                print("-"*80)
                
                await ws.send(json.dumps({"inputText": prompt}))
                response = await ws.recv()
                result = json.loads(response)
                print(f"Response: {result.get('result', result)}")
                
                if i < len(test_prompts):
                    await asyncio.sleep(0.5)
            
            print("\n" + "="*80)
            print("Testing Complete!")
            print("="*80)
            
    except websockets.exceptions.InvalidStatus as e:
        print(f"WebSocket handshake failed with status code: {e.response.status_code}")
        print(f"Response headers: {e.response.headers}")
        print(f"Response body: {e.response.body.decode()}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
