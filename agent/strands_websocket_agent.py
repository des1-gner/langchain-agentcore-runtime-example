from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
import datetime
import random
import hashlib
import uuid

app = BedrockAgentCoreApp()

# Define simple tools as functions
def get_current_timestamp() -> str:
    """Get the current exact timestamp in ISO format."""
    return datetime.datetime.now().isoformat()

def generate_random_number(min_val: int, max_val: int) -> int:
    """Generate a random number between min_val and max_val."""
    return random.randint(min_val, max_val)

def generate_uuid() -> str:
    """Generate a unique UUID."""
    return str(uuid.uuid4())

def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Hash a string using the specified algorithm (md5, sha1, sha256, sha512)."""
    text_bytes = text.encode('utf-8')
    
    if algorithm == "md5":
        return hashlib.md5(text_bytes).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text_bytes).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text_bytes).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(text_bytes).hexdigest()
    else:
        return f"Unsupported algorithm: {algorithm}"

# Initialize agent with tools
agent = Agent(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    tools=[
        get_current_timestamp,
        generate_random_number,
        generate_uuid,
        hash_string
    ]
)

@app.websocket
async def websocket_handler(websocket, context):
    """WebSocket handler for bidirectional streaming with Strands agent"""
    print("Awaiting websocket connection")
    await websocket.accept()
    
    print("WebSocket connected")
    await websocket.send_json({"type": "connected"})
    
    try:
        while True:
            data = await websocket.receive_json()
            user_text = data.get("prompt", "")
            session_id = data.get("session_id", "default")
            
            if not user_text:
                continue
            
            print(f"[Session: {session_id}] Received: {user_text}")
            
            # Stream response from agent
            async for chunk in agent.stream_async(user_text, session_id=session_id):
                if isinstance(chunk, str):
                    await websocket.send_json({"type": "token", "content": chunk})
            
            await websocket.send_json({"type": "done"})
            
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
        print("WebSocket closed")

if __name__ == "__main__":
    app.run(log_level="info")
