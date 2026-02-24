# Deploy Strands WebSocket Agent

This guide shows how to deploy a WebSocket agent using the Strands framework with streaming responses.

## What's Different?

**Strands Agent Features:**
- Built-in streaming support with `stream_async()`
- Session management for conversation continuity
- Simpler tool integration (just pass functions)
- Token-by-token streaming responses

## Step 1: Install Dependencies

```bash
cd agent
source .venv/bin/activate
pip install strands
```

## Step 2: Test Locally (Optional)

```bash
python strands_websocket_agent.py
```

In another terminal:
```bash
cd client
python3 websocket_client_strands.py
```

## Step 3: Configure for Deployment

```bash
cd agent
source .venv/bin/activate
agentcore configure -e strands_websocket_agent.py
```

When prompted for dependency file, specify:
```
strands_requirements.txt
```

## Step 4: Deploy

```bash
agentcore deploy
```

Save the Agent ARN from the output.

## Step 5: Test Deployed Agent

```bash
export AGENT_ARN="your-agent-arn-here"
cd client
python3 websocket_client_strands.py
```

## Key Features

### Streaming Responses
The agent streams tokens as they're generated:
```python
async for chunk in agent.stream_async(user_text, session_id=session_id):
    if isinstance(chunk, str):
        await websocket.send_json({"type": "token", "content": chunk})
```

### Session Management
Maintains conversation context across messages:
```python
session_id = data.get("session_id", "default")
agent.stream_async(user_text, session_id=session_id)
```

### Simple Tool Integration
Just pass Python functions as tools:
```python
agent = Agent(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    tools=[
        get_current_timestamp,
        generate_random_number,
        generate_uuid,
        hash_string
    ]
)
```

## Message Protocol

**Client → Server:**
```json
{
    "prompt": "What is the current time?",
    "session_id": "user-123"
}
```

**Server → Client (streaming):**
```json
{"type": "connected"}
{"type": "token", "content": "The"}
{"type": "token", "content": " current"}
{"type": "token", "content": " time"}
{"type": "token", "content": " is..."}
{"type": "done"}
```

**Server → Client (error):**
```json
{"type": "error", "message": "Error description"}
```

## Comparison: LangChain vs Strands

| Feature | LangChain Agent | Strands Agent |
|---------|----------------|---------------|
| Streaming | Manual implementation | Built-in `stream_async()` |
| Tools | Decorator-based `@tool` | Simple functions |
| Sessions | Manual tracking | Built-in session management |
| Response | Single complete response | Token-by-token streaming |
| Complexity | More verbose | More concise |

## Next Steps

- Add more custom tools
- Implement conversation memory
- Add error handling and retries
- Monitor with CloudWatch logs

## Troubleshooting

### Import Error: strands
```bash
pip install strands
```

### Model Access Error
Ensure you have access to Claude 3.5 Sonnet in Amazon Bedrock console.

### Session Not Persisting
Sessions are maintained in-memory. For persistent sessions, integrate with Amazon Bedrock AgentCore Memory.
