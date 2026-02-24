#!/bin/bash

export AGENT_ARN="arn:aws:bedrock-agentcore:us-east-1:331135961154:runtime/websocket_agent-kLERaH7G2V"

echo "Testing WebSocket Agent"
echo "ARN: $AGENT_ARN"
echo ""

cd client
python3 websocket_client_sigv4_headers.py
