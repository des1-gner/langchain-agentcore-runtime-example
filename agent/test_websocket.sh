#!/bin/bash

# Make sure AGENT_ARN is set
if [ -z "$AGENT_ARN" ]; then
    echo "Error: AGENT_ARN environment variable is not set"
    echo "Usage: export AGENT_ARN='your-agent-arn'"
    exit 1
fi

echo "Testing WebSocket agent with ARN: $AGENT_ARN"
echo ""

cd client
python websocket_client_sigv4_headers.py
