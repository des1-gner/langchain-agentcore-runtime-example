# LangChain WebSocket Agent on Amazon Bedrock AgentCore Runtime

A collection of WebSocket-enabled AI agents deployed on Amazon Bedrock AgentCore Runtime, demonstrating bidirectional streaming with LangChain and custom tools.

## 🚀 Features

- **WebSocket Bidirectional Streaming** - Real-time communication between client and agent
- **Multiple Agent Implementations**:
  - Basic WebSocket agent with LangChain
  - Streaming WebSocket agent with token-by-token responses
  - Strands framework integration (optional)
- **Custom Tools** - Timestamp, random numbers, UUIDs, hashing, date calculations
- **Session Management** - Conversation context across multiple messages
- **Multiple Authentication Methods** - SigV4 headers, pre-signed URLs, OAuth
- **Production Ready** - Deployed on AWS with CloudWatch logging and observability

## 📋 Prerequisites

- Python 3.10 or higher
- AWS account with credentials configured
- AWS CLI installed and configured
- Model access: Anthropic Claude Sonnet 4.5 in Amazon Bedrock console

## 🏗️ Project Structure

```
.
├── agent/
│   ├── websocket_agent.py                    # Basic WebSocket agent
│   ├── langchain_streaming_websocket_agent.py # Streaming agent
│   ├── strands_websocket_agent.py            # Strands framework agent
│   ├── requirements.txt                       # Agent dependencies
│   └── strands_requirements.txt              # Strands-specific dependencies
├── client/
│   ├── websocket_client_local.py             # Local testing client
│   ├── websocket_client_sigv4_headers.py     # SigV4 headers auth
│   ├── websocket_client_sigv4_presigned_url.py # SigV4 pre-signed URL
│   ├── websocket_client_oauth.py             # OAuth authentication
│   ├── websocket_client_strands.py           # Strands agent client
│   └── requirements.txt                       # Client dependencies
├── QUICKSTART.md                              # Quick deployment guide
├── README_WEBSOCKET.md                        # Detailed WebSocket documentation
├── DEPLOY_STRANDS.md                          # Strands deployment guide
└── README.md                                  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd agent
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Test Locally (Optional)

```bash
python websocket_agent.py
```

In another terminal:
```bash
cd client
pip install -r requirements.txt
python websocket_client_local.py
```

### 3. Deploy to AWS

```bash
cd agent
source .venv/bin/activate
agentcore configure -e websocket_agent.py
agentcore deploy
```

Save the Agent ARN from the output.

### 4. Test Deployed Agent

```bash
export AGENT_ARN="your-agent-arn-here"
cd client
python websocket_client_sigv4_headers.py
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step deployment guide
- **[README_WEBSOCKET.md](README_WEBSOCKET.md)** - Comprehensive WebSocket documentation
- **[DEPLOY_STRANDS.md](DEPLOY_STRANDS.md)** - Strands framework deployment

## 🔧 Available Agents

### 1. Basic WebSocket Agent (`websocket_agent.py`)
- Simple request-response over WebSocket
- LangChain with tool calling
- Complete responses (not streaming)

### 2. Streaming WebSocket Agent (`langchain_streaming_websocket_agent.py`)
- Token-by-token streaming responses
- Session management with conversation history
- Real-time response generation

### 3. Strands Agent (`strands_websocket_agent.py`)
- Built on Strands framework
- Simplified tool integration
- Advanced streaming capabilities

## 🛠️ Custom Tools

All agents include these custom tools:

- **get_current_timestamp()** - Returns current ISO timestamp
- **generate_random_number(min, max)** - Generates random numbers
- **generate_uuid()** - Creates unique UUIDs
- **hash_string(text, algorithm)** - Computes cryptographic hashes
- **calculate_file_size(bytes)** - Converts bytes to human-readable format
- **get_day_of_week(year, month, day)** - Calculates day of week
- **calculate_days_between(start, end)** - Computes date differences

## 🔐 Authentication Methods

### SigV4 Headers (Recommended)
```python
ws_url, headers = client.generate_ws_connection(runtime_arn=runtime_arn)
async with websockets.connect(ws_url, additional_headers=headers) as ws:
    # Your code here
```

### SigV4 Pre-signed URL
```python
sigv4_url = client.generate_presigned_url(runtime_arn=runtime_arn, expires=300)
async with websockets.connect(sigv4_url) as ws:
    # Your code here
```

### OAuth
```python
ws_url, headers = client.generate_ws_connection_oauth(
    runtime_arn=runtime_arn,
    bearer_token=bearer_token
)
async with websockets.connect(ws_url, additional_headers=headers) as ws:
    # Your code here
```

## 📊 Message Protocol

### Client → Server
```json
{
    "prompt": "What is the current timestamp?",
    "session_id": "user-123"
}
```

### Server → Client (Basic Agent)
```json
{
    "result": "The current timestamp is 2026-02-24T04:45:42.724893"
}
```

### Server → Client (Streaming Agent)
```json
{"type": "connected"}
{"type": "token", "content": "The"}
{"type": "token", "content": " current"}
{"type": "token", "content": " timestamp"}
{"type": "done"}
```

## 🔍 Monitoring & Observability

### CloudWatch Logs
```bash
aws logs tail /aws/bedrock-agentcore/runtimes/YOUR-AGENT-ID-DEFAULT \
  --log-stream-name-prefix "2026/02/24/[runtime-logs" \
  --follow
```

### GenAI Observability Dashboard
Visit the AWS Console:
```
https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#gen-ai-observability/agent-core
```

### Check Agent Status
```bash
cd agent
source .venv/bin/activate
agentcore status
```

## 🆚 WebSocket vs HTTP

| Feature | HTTP Agent | WebSocket Agent |
|---------|-----------|-----------------|
| Connection | Request-response | Persistent bidirectional |
| Streaming | No | Yes (token-by-token) |
| Latency | Higher | Lower |
| Use Case | Simple queries | Real-time conversations |
| Interruption | Not supported | Supported |

## 🧪 Testing

### Test All Authentication Methods
```bash
export AGENT_ARN="your-agent-arn"

# SigV4 Headers
python client/websocket_client_sigv4_headers.py

# SigV4 Pre-signed URL
python client/websocket_client_sigv4_presigned_url.py

# OAuth (requires BEARER_TOKEN)
export BEARER_TOKEN="your-token"
python client/websocket_client_oauth.py
```

## 🐛 Troubleshooting

### SSL Certificate Error (macOS)
```bash
pip install --upgrade certifi
```

### AWS Credentials Error
```bash
aws sts get-caller-identity
aws configure
```

### Module Not Found: strands
```bash
pip install strands-agents
```

### Connection Refused (Local Testing)
Make sure the agent is running:
```bash
python agent/websocket_agent.py
```

## 📦 Deployment Options

### Direct Code Deploy (Recommended)
- No Docker required
- Faster deployment
- Python 3.10, 3.11, 3.12, 3.13 support

### Container Deploy
- Custom runtimes
- Complex dependencies
- Full control over environment

## 🔄 Session Management

Sessions maintain conversation context:

```python
# Client sends session_id
await ws.send(json.dumps({
    "prompt": "Remember this: my name is Alice",
    "session_id": "user-123"
}))

# Later in the same session
await ws.send(json.dumps({
    "prompt": "What's my name?",
    "session_id": "user-123"
}))
# Response: "Your name is Alice"
```

## 🧹 Clean Up

```bash
cd agent
source .venv/bin/activate
agentcore destroy
```

Or manually delete resources in AWS Console:
- CloudWatch Log Groups
- IAM Roles (AmazonBedrockAgentCoreSDKRuntime-*)
- S3 Buckets (bedrock-agentcore-*)
- AgentCore Runtimes

## 📖 Additional Resources

- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- [LangChain Documentation](https://python.langchain.com/)
- [WebSocket RFC 6455](https://tools.ietf.org/html/rfc6455)
- [Strands Framework](https://github.com/awslabs/strands)

## 🤝 Contributing

This is a private repository. For questions or issues, contact the repository owner.

## 📄 License

Private - All rights reserved.

## 🙏 Acknowledgments

- Amazon Bedrock AgentCore team
- LangChain community
- Strands framework developers
