# Quick Start Guide - WebSocket Agent

## Step 1: Activate Virtual Environment

```bash
cd agent
source .venv/bin/activate
```

## Step 2: Test Locally (Optional)

Start the WebSocket agent:
```bash
python websocket_agent.py
```

In another terminal, test it:
```bash
cd client
source ../agent/.venv/bin/activate
pip install -r requirements.txt
python websocket_client_local.py
```

Press `Ctrl+C` to stop the agent.

## Step 3: Configure for Deployment

```bash
cd agent
source .venv/bin/activate
agentcore configure -e websocket_agent.py
```

Follow the prompts:
- **Execution Role**: Press Enter (auto-create)
- **ECR Repository**: Press Enter (auto-create)
- **Dependency file**: Press Enter (use requirements.txt)
- **Authorization**: Choose based on your needs
- **Other settings**: Press Enter to accept defaults

## Step 4: Deploy to AWS

```bash
agentcore deploy
```

This will:
1. Build a Docker container
2. Push to ECR
3. Deploy to AgentCore Runtime
4. Return your Agent ARN

Save the Agent ARN - you'll need it!

## Step 5: Set Environment Variable

```bash
export AGENT_ARN="arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/websocket-agent-xyz"
```

Replace with your actual ARN from step 4.

## Step 6: Test Deployed Agent

### Option A: SigV4 Headers (Recommended)
```bash
cd client
python websocket_client_sigv4_headers.py
```

### Option B: SigV4 Pre-signed URL
```bash
python websocket_client_sigv4_presigned_url.py
```

### Option C: OAuth (if configured)
```bash
export BEARER_TOKEN="your_token_here"
python websocket_client_oauth.py
```

## Troubleshooting

### Command not found: agentcore
Make sure you've activated the virtual environment:
```bash
cd agent
source .venv/bin/activate
```

Or use the full path:
```bash
.venv/bin/agentcore --help
```

### AWS Credentials
Verify your AWS credentials are configured:
```bash
aws sts get-caller-identity
```

### Region Mismatch
Update the region in client files if needed:
```python
client = AgentCoreRuntimeClient(region="us-east-1")  # Change to your region
```

## What's Different from HTTP?

1. **Agent**: Uses `@app.websocket` instead of `@app.entrypoint`
2. **Connection**: Persistent bidirectional connection vs request-response
3. **Client**: Uses `websockets` library with async/await
4. **Authentication**: Same methods (SigV4, OAuth) but applied to WebSocket handshake

## Next Steps

- Read `README_WEBSOCKET.md` for detailed documentation
- Check AWS Console for logs: CloudWatch → `/aws/bedrock-agentcore/runtimes/`
- Try session management for conversation continuity
- Explore custom headers for passing context

## Clean Up

When done, destroy resources:
```bash
cd agent
source .venv/bin/activate
agentcore destroy
```
