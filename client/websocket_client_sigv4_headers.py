from bedrock_agentcore.runtime import AgentCoreRuntimeClient
import websockets
import asyncio
import json
import os
import ssl
import certifi

async def main():
    """Test WebSocket connection using SigV4 signed headers"""
    # Get runtime ARN from environment variable
    runtime_arn = os.getenv('AGENT_ARN')
    if not runtime_arn:
        raise ValueError("AGENT_ARN environment variable is required")
    
    # Extract region from ARN
    region = runtime_arn.split(':')[3]
    
    # Initialize client
    client = AgentCoreRuntimeClient(region=region)
    
    # Generate WebSocket connection with authentication
    ws_url, headers = client.generate_ws_connection(
        runtime_arn=runtime_arn
    )
    
    test_prompts = [
        "What is the current timestamp?",
        "Generate a random number between 1 and 1000",
        "Generate a UUID for me",
        "What is the SHA256 hash of the word 'hello'?",
        "Convert 1073741824 bytes to human readable format",
        "What day of the week was January 1, 2000?",
        "How many days between 2020-01-01 and 2025-12-31?",
    ]
    
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    try:
        async with websockets.connect(ws_url, additional_headers=headers, ssl=ssl_context) as ws:
            print("="*80)
            print("Testing WebSocket Agent with SigV4 Headers")
            print("="*80)
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\nTest {i}/{len(test_prompts)}: {prompt}")
                print("-"*80)
                
                # Send message
                await ws.send(json.dumps({"inputText": prompt}))
                
                # Receive response
                response = await ws.recv()
                result = json.loads(response)
                print(f"Response: {result.get('result', result)}")
                
                # Small delay between requests
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
