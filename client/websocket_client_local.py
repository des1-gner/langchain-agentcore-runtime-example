import asyncio
import websockets
import json

async def local_websocket():
    """Test WebSocket connection to local agent"""
    uri = "ws://localhost:8080/ws"
    
    test_prompts = [
        "What is the current timestamp?",
        "Generate a random number between 1 and 1000",
        "Generate a UUID for me",
        "What is the SHA256 hash of the word 'hello'?",
    ]
    
    try:
        async with websockets.connect(uri) as websocket:
            print("="*80)
            print("Testing Local WebSocket Agent")
            print("="*80)
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\nTest {i}/{len(test_prompts)}: {prompt}")
                print("-"*80)
                
                # Send message
                await websocket.send(json.dumps({"inputText": prompt}))
                
                # Receive response
                response = await websocket.recv()
                result = json.loads(response)
                print(f"Response: {result.get('result', result)}")
                
                # Small delay between requests
                if i < len(test_prompts):
                    await asyncio.sleep(0.5)
            
            print("\n" + "="*80)
            print("Testing Complete!")
            print("="*80)
            
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(local_websocket())
