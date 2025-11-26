import boto3
import json
import uuid
import time

# Replace with your actual Agent ARN
agent_arn = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/langchain-agent-abc123"

# Initialize the Bedrock AgentCore client
agentcore_client = boto3.client('bedrock-agentcore', region_name='us-east-1')

# Test prompts that prove the tools are being used (LLMs cannot do these)
test_prompts = [
    "What is the current timestamp?",
    "Generate a random number between 1 and 1000",
    "Generate a UUID for me",
    "What is the SHA256 hash of the word 'hello'?",
    "Convert 1073741824 bytes to human readable format",
    "What day of the week was January 1, 2000?",
    "How many days between 2020-01-01 and 2025-12-31?",
]

print("="*80)
print("Testing LangChain Agent with Tools on Amazon Bedrock AgentCore Runtime")
print("="*80)
print("\nThese tools prove the agent is actually executing code:")
print("- Current timestamp (LLM training data is frozen)")
print("- Random numbers (LLM cannot generate true randomness)")
print("- UUIDs (LLM cannot create valid UUIDs)")
print("- Cryptographic hashes (LLM cannot compute actual hashes)")
print("- Date calculations (LLM cannot accurately compute date arithmetic)")
print("="*80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*80}")
    print(f"Test {i}/{len(test_prompts)}: {prompt}")
    print(f"{'='*80}")
    
    # Prepare the payload
    payload = json.dumps({"prompt": prompt}).encode()
    
    try:
        # Invoke the agent
        response = agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=str(uuid.uuid4()),
            payload=payload,
            qualifier="DEFAULT"
        )
        
        # Process the response
        content = []
        for chunk in response.get("response", []):
            content.append(chunk.decode('utf-8'))
        
        result = json.loads(''.join(content))
        print(f"Result: {result['result']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Small delay between requests
    if i < len(test_prompts):
        time.sleep(1)

print("\n" + "="*80)
print("Testing Complete!")
print("="*80)

# Additional verification test: Call timestamp twice to prove it's dynamic
print("\n" + "="*80)
print("VERIFICATION TEST: Calling timestamp tool twice to prove it's dynamic")
print("="*80)

for attempt in [1, 2]:
    print(f"\nAttempt {attempt}:")
    payload = json.dumps({"prompt": "What is the current timestamp?"}).encode()
    
    try:
        response = agentcore_client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=str(uuid.uuid4()),
            payload=payload,
            qualifier="DEFAULT"
        )
        
        content = []
        for chunk in response.get("response", []):
            content.append(chunk.decode('utf-8'))
        
        result = json.loads(''.join(content))
        print(f"  Timestamp: {result['result']}")
        
        if attempt == 1:
            print("  Waiting 3 seconds...")
            time.sleep(3)
            
    except Exception as e:
        print(f"  Error: {str(e)}")

print("\n" + "="*80)
print("If the timestamps are different, the tool is definitely being called!")
print("="*80)
