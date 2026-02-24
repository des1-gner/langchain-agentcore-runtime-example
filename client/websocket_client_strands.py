from bedrock_agentcore.runtime import AgentCoreRuntimeClient
import websockets
import asyncio
import json
import os
import ssl
import certifi

async def main():
    """Test WebSocket connection with Strands streaming agent"""
    runtime_arn = os.getenv('AGENT_ARN')
    if not runtime_arn:
        raise ValueError("AGENT_ARN environment variable is required")
    
    # Extract region from ARN
    region = runtime_arn.split(':')[3]
    
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
    ]
    
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    try:
        async with websockets.connect(ws_url, additional_headers=headers, ssl=ssl_context) as ws:
            print("="*80)
            print("Testing Strands WebSocket Agent with Streaming")
            print("="*80)
            
            # Wait for connection confirmation
            connect_msg = await ws.recv()
            print(f"Connection: {connect_msg}\n")
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"Test {i}/{len(test_prompts)}: {prompt}")
                print("-"*80)
                
                # Send message
                await ws.send(json.dumps({"prompt": prompt, "session_id": "test-session"}))
                
                # Receive streaming response
                response_text = ""
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "token":
                        token = data.get("content", "")
                        response_text += token
                        print(token, end="", flush=True)
                    elif data.get("type") == "done":
                        print("\n")
                        break
                    elif data.get("type") == "error":
                        print(f"\nError: {data.get('message')}")
                        break
                
                if i < len(test_prompts):
                    await asyncio.sleep(0.5)
            
            print("="*80)
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
