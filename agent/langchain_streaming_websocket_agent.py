from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from bedrock_agentcore import BedrockAgentCoreApp
import datetime
import random
import hashlib
import uuid

app = BedrockAgentCoreApp()

# Define custom tools
@tool
def get_current_timestamp() -> str:
    """Get the current exact timestamp in ISO format."""
    return datetime.datetime.now().isoformat()

@tool
def generate_random_number(min_val: int, max_val: int) -> int:
    """Generate a random number between min_val and max_val."""
    return random.randint(min_val, max_val)

@tool
def generate_uuid() -> str:
    """Generate a unique UUID."""
    return str(uuid.uuid4())

@tool
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

# Initialize the LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={"temperature": 0.1},
    streaming=True
)

# Bind tools to the LLM
tools = [
    get_current_timestamp,
    generate_random_number,
    generate_uuid,
    hash_string
]
llm_with_tools = llm.bind_tools(tools)

# Session storage (in-memory)
sessions = {}

@app.websocket
async def websocket_handler(websocket, context):
    """WebSocket handler for bidirectional streaming with LangChain"""
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
            
            # Get or create session history
            if session_id not in sessions:
                sessions[session_id] = [
                    SystemMessage(content="""You are a helpful assistant with access to tools. 
You MUST use the available tools when asked about:
- Current time/timestamp: use get_current_timestamp
- Random numbers: use generate_random_number
- UUIDs: use generate_uuid
- Hashing: use hash_string

Always provide the tool result in your response.""")
                ]
            
            messages = sessions[session_id].copy()
            messages.append(HumanMessage(content=user_text))
            
            # First invocation - check if tools are needed
            response = llm_with_tools.invoke(messages)
            
            # Check if the model wants to use tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"[Session: {session_id}] Tool calls requested: {len(response.tool_calls)}")
                messages.append(response)
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call.get("id", "")
                    
                    print(f"[Session: {session_id}] Executing tool: {tool_name}")
                    
                    # Find and execute the tool
                    tool_result = None
                    for tool_func in tools:
                        if tool_func.name == tool_name:
                            tool_result = tool_func.invoke(tool_args)
                            break
                    
                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"
                    
                    messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
                
                # Stream final response with tool results
                print(f"[Session: {session_id}] Streaming final response...")
                async for chunk in llm_with_tools.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        await websocket.send_json({"type": "token", "content": chunk.content})
                
                # Update session history
                final_response = llm_with_tools.invoke(messages)
                sessions[session_id].append(HumanMessage(content=user_text))
                sessions[session_id].append(final_response)
            else:
                # No tools needed, stream direct response
                print(f"[Session: {session_id}] Streaming direct response...")
                async for chunk in llm_with_tools.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        await websocket.send_json({"type": "token", "content": chunk.content})
                
                # Update session history
                sessions[session_id].append(HumanMessage(content=user_text))
                sessions[session_id].append(response)
            
            await websocket.send_json({"type": "done"})
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
        print("WebSocket closed")

if __name__ == "__main__":
    app.run(log_level="info")
