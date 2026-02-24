# LangChain Agent with WebSocket Bidirectional Streaming

This guide demonstrates how to deploy a LangChain agent with WebSocket bidirectional streaming support on Amazon Bedrock AgentCore Runtime.

## Overview

**Information** | **Details**
--- | ---
Agent type | Bidirectional Streaming (WebSocket)
Framework | LangChain
LLM model | Anthropic Claude Sonnet 4.5
Components | AgentCore Runtime with WebSocket
Complexity | Intermediate
SDK used | Amazon Bedrock AgentCore Python SDK

This example shows how to create a LangChain agent with custom tools that supports real-time bidirectional communication using WebSocket connections.

## Prerequisites

- Python 3.10 or higher
- AWS account with appropriate permissions
- AWS CLI configured with credentials
- Model access: Anthropic Claude Sonnet 4.5 enabled in Amazon Bedrock console

## Project Structure

```
langchain-agentcore-runtime/
├── agent/
│   ├── langchain_agent.py           # Original HTTP agent
│   ├── websocket_agent.py           # WebSocket agent implementation
│   └── requirements.txt              # Agent dependencies
├── client/
│   ├── test_client.py               # Original HTTP client
│   ├── websocket_client_local.py    # Local WebSocket testing
│   ├── websocket_client_sigv4_headers.py      # SigV4 headers auth
│   ├── websocket_client_sigv4_presigned_url.py # SigV4 presigned URL
│   ├── websocket_client_oauth.py    # OAuth authentication
│   └── requirements.txt              # Client dependencies
├── README.md                         # Original README
└── README_WEBSOCKET.md              # This file
```

## Step 1: Install Dependencies

Navigate to the agent directory and install dependencies:

```bash
cd agent
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Test WebSocket Agent Locally

Start your WebSocket agent locally:

```bash
python websocket_agent.py
```

You should see output indicating the server is running on port 8080.

In another terminal, test the local WebSocket connection:

```bash
cd ../client
source ../agent/.venv/bin/activate  # Use the same virtual environment
pip install -r requirements.txt
python websocket_client_local.py
```

Press `Ctrl+C` in the agent terminal to stop the local agent when done testing.

## Step 3: Deploy WebSocket Agent to AWS

Configure your WebSocket agent for deployment:

```bash
cd ../agent
agentcore configure -e websocket_agent.py
```

During configuration:
- **Execution Role**: Press Enter to auto-create
- **ECR Repository**: Press Enter to auto-create
- **Dependency file**: Press Enter to use detected `requirements.txt`
- **Authorization**: Choose based on your needs (OAuth or AWS credentials)

Deploy to AWS:

```bash
agentcore launch
```

After successful deployment, you'll receive an **Agent ARN** like:

```
arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/websocket-agent-abc123
```

**Save this ARN** - you'll need it for testing.

## Step 4: Set Environment Variables

Export your agent ARN:

```bash
export AGENT_ARN="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/websocket-agent-abc123"
```

If using OAuth, also export your bearer token:

```bash
export BEARER_TOKEN="your_oauth_token_here"
```

## Step 5: Test Deployed WebSocket Agent

### Option 1: SigV4 Signed Headers (Recommended)

```bash
cd ../client
python websocket_client_sigv4_headers.py
```

### Option 2: SigV4 Pre-signed URL

```bash
python websocket_client_sigv4_presigned_url.py
```

### Option 3: OAuth Authentication

```bash
python websocket_client_oauth.py
```

## Authentication Methods

The WebSocket implementation supports three authentication methods:

1. **AWS Signature Version 4 headers**: Sign the WebSocket handshake request headers using your AWS credentials
2. **AWS Signature Version 4 Pre-signed URL**: Create a presigned WebSocket URL with SigV4 signature as query parameters
3. **OAuth Bearer token**: Pass an OAuth token in the Authorization header

## Key Differences from HTTP Agent

### Agent Implementation

**HTTP Agent (`langchain_agent.py`)**:
```python
@app.entrypoint
def invoke_agent(payload, context=None):
    # Synchronous request-response
    user_input = payload.get("prompt", "Hello!")
    response = process_message(user_input)
    return {"result": response}
```

**WebSocket Agent (`websocket_agent.py`)**:
```python
@app.websocket
async def websocket_handler(websocket, context):
    # Bidirectional streaming
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        response = process_message(data.get("inputText", ""))
        await websocket.send_json({"result": response})
```

### Client Implementation

**HTTP Client**:
```python
response = agentcore_client.invoke_agent_runtime(
    agentRuntimeArn=agent_arn,
    runtimeSessionId=str(uuid.uuid4()),
    payload=payload
)
```

**WebSocket Client**:
```python
async with websockets.connect(ws_url, additional_headers=headers) as ws:
    await ws.send(json.dumps({"inputText": prompt}))
    response = await ws.recv()
```

## Session Management

To use sessions with WebSocket connections, pass a session ID when establishing the connection:

```python
ws_url, headers = client.generate_ws_connection(
    runtime_arn=runtime_arn,
    session_id="user-123-conversation-456"
)
```

Sessions provide:
- Conversation context across multiple connections
- Isolated execution environments
- Automatic idle timeout reset on message activity

## Expected Output

When you run any of the WebSocket clients, you should see output like:

```
================================================================================
Testing WebSocket Agent with SigV4 Headers
================================================================================

Test 1/7: What is the current timestamp?
--------------------------------------------------------------------------------
Response: The current timestamp is **2025-11-26T05:40:05.317229** (in ISO format).

Test 2/7: Generate a random number between 1 and 1000
--------------------------------------------------------------------------------
Response: The random number generated between 1 and 1000 is **718**.

Test 3/7: Generate a UUID for me
--------------------------------------------------------------------------------
Response: I've generated a UUID for you: **fd45926a-f5c5-4173-8159-4e561d71ae98**

...

================================================================================
Testing Complete!
================================================================================
```

## When to Use WebSocket vs HTTP

**Use WebSocket when:**
- Real-time voice conversations with immediate audio streaming
- Bidirectional data flow (streaming from client to agent and vice versa)
- Interrupt handling (user can interrupt agent mid-conversation)
- Long-lived connections with multiple message exchanges

**Use HTTP when:**
- Simple request-response patterns
- No bidirectional streaming needed
- Stateless interactions

## Troubleshooting

### Connection Failures
- Verify agent processes connections at `/ws`
- Check authentication method matches agent configuration

### Message Frame Size Exceeded
- Configure message fragmentation for large messages
- Split messages into chunks below 32KB limit

### Health Check Failures
- Ensure agent implements `/ping` endpoint
- Check CloudWatch logs for errors

## Additional Resources

- [WebSocket RFC 6455](https://tools.ietf.org/html/rfc6455)
- [Amazon Bedrock AgentCore WebSocket Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/websocket-streaming.html)
- [WebSocket Bidirectional Streaming Samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
- [LangChain Documentation](https://python.langchain.com/)

## Clean Up

When you're done, delete the AWS resources using the AWS Console or CLI.
