# LangChain Agent on Amazon Bedrock AgentCore Runtime

This guide demonstrates how to deploy a LangChain agent with custom tools to Amazon Bedrock AgentCore Runtime.

## Overview

**Information** | **Details**
--- | ---
Agent type | Synchronous
Framework | LangChain
LLM model | Anthropic Claude Sonnet 4.5
Components | AgentCore Runtime
Complexity | Easy
SDK used | Amazon Bedrock AgentCore Python SDK

This example shows how to create a LangChain agent with custom tools that an LLM cannot perform on its own (timestamps, cryptographic hashes, random numbers, date calculations) and deploy it as a managed service on AWS.

## Prerequisites

- Python 3.10 or higher
- AWS account with appropriate permissions
- AWS CLI configured with credentials
- Boto3 installed
- Model access: Anthropic Claude Sonnet 4.5 enabled in Amazon Bedrock console

## Project Structure

```
langchain-agentcore-runtime/
├── agent/
│   ├── langchain_agent.py      # Agent implementation
│   └── requirements.txt         # Agent dependencies
├── client/
│   ├── test_client.py          # Test client
│   └── requirements.txt         # Client dependencies
└── README.md                    # This file
```

**Note:** If you're cloning this repository, you can skip to Step 3 as the files are already created.

## Step 1: Create Your LangChain Agent

Create `agent/langchain_agent.py`:

```python
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from bedrock_agentcore import BedrockAgentCoreApp
import datetime
import random
import hashlib
import uuid
import json

app = BedrockAgentCoreApp()

# Define custom tools that LLMs cannot do
@tool
def get_current_timestamp() -> str:
    """Get the current exact timestamp in ISO format. LLMs cannot know the current time."""
    timestamp = datetime.datetime.now().isoformat()
    print(f"[TOOL] get_current_timestamp called, returning: {timestamp}")
    return timestamp

@tool
def generate_random_number(min_val: int, max_val: int) -> int:
    """Generate a truly random number between min_val and max_val. LLMs cannot generate true randomness."""
    result = random.randint(min_val, max_val)
    print(f"[TOOL] generate_random_number called with min={min_val}, max={max_val}, returning: {result}")
    return result

@tool
def generate_uuid() -> str:
    """Generate a unique UUID. LLMs cannot generate true UUIDs."""
    result = str(uuid.uuid4())
    print(f"[TOOL] generate_uuid called, returning: {result}")
    return result

@tool
def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    Hash a string using the specified algorithm (md5, sha1, sha256, sha512).
    LLMs cannot compute actual cryptographic hashes.
    """
    text_bytes = text.encode('utf-8')
    
    if algorithm == "md5":
        result = hashlib.md5(text_bytes).hexdigest()
    elif algorithm == "sha1":
        result = hashlib.sha1(text_bytes).hexdigest()
    elif algorithm == "sha256":
        result = hashlib.sha256(text_bytes).hexdigest()
    elif algorithm == "sha512":
        result = hashlib.sha512(text_bytes).hexdigest()
    else:
        result = f"Unsupported algorithm: {algorithm}. Use md5, sha1, sha256, or sha512."
    
    print(f"[TOOL] hash_string called with text='{text}', algorithm='{algorithm}', returning: {result}")
    return result

@tool
def calculate_file_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format (KB, MB, GB, TB).
    Returns exact conversions that require precise computation.
    """
    units = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    result = f"{round(size, 2)} {units[unit_index]}"
    print(f"[TOOL] calculate_file_size called with {size_bytes} bytes, returning: {result}")
    return result

@tool
def get_day_of_week(year: int, month: int, day: int) -> str:
    """
    Calculate the exact day of the week for any date.
    LLMs cannot accurately compute this for arbitrary dates.
    """
    try:
        date_obj = datetime.date(year, month, day)
        day_name = date_obj.strftime("%A")
        result = f"{year}-{month:02d}-{day:02d} was/is a {day_name}"
        print(f"[TOOL] get_day_of_week called with {year}-{month}-{day}, returning: {result}")
        return result
    except ValueError as e:
        error_msg = f"Invalid date: {str(e)}"
        print(f"[TOOL] get_day_of_week error: {error_msg}")
        return error_msg

@tool
def calculate_days_between(start_date: str, end_date: str) -> str:
    """
    Calculate exact number of days between two dates (format: YYYY-MM-DD).
    LLMs cannot accurately compute date differences.
    """
    try:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        difference = end - start
        
        result = f"{difference.days} days (approximately {round(difference.days / 7, 2)} weeks or {round(difference.days / 365.25, 2)} years)"
        print(f"[TOOL] calculate_days_between called with {start_date} to {end_date}, returning: {result}")
        return result
    except ValueError as e:
        error_msg = f"Invalid date format. Use YYYY-MM-DD. Error: {str(e)}"
        print(f"[TOOL] calculate_days_between error: {error_msg}")
        return error_msg

# Initialize the LLM
llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={"temperature": 0.1}
)

# Bind tools to the LLM
tools = [
    get_current_timestamp,
    generate_random_number,
    generate_uuid,
    hash_string,
    calculate_file_size,
    get_day_of_week,
    calculate_days_between
]
llm_with_tools = llm.bind_tools(tools)

@app.entrypoint
def invoke_agent(payload, context=None):
    """
    Entry point for the agent invocation
    """
    try:
        user_input = payload.get("prompt", "Hello!")
        print(f"[DEBUG] Received prompt: {user_input}")
        
        # Create initial messages with system message
        messages = [
            SystemMessage(content="""You are a helpful assistant with access to tools. 
You MUST use the available tools when asked about:
- Current time/timestamp: use get_current_timestamp
- Random numbers: use generate_random_number
- UUIDs: use generate_uuid
- Hashing: use hash_string
- File size conversions: use calculate_file_size
- Day of week: use get_day_of_week
- Date calculations: use calculate_days_between

Always provide the tool result in your response."""),
            HumanMessage(content=user_input)
        ]
        
        # First invocation - let the model decide if it needs tools
        print("[DEBUG] Invoking LLM with tools...")
        response = llm_with_tools.invoke(messages)
        print(f"[DEBUG] LLM response received. Has tool_calls: {hasattr(response, 'tool_calls') and bool(response.tool_calls)}")
        
        # Check if the model wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[DEBUG] Tool calls requested: {len(response.tool_calls)}")
            # Add the AI response to messages
            messages.append(response)
            
            # Execute each tool call and add results
            for i, tool_call in enumerate(response.tool_calls):
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call.get("id", "")
                
                print(f"[DEBUG] Executing tool {i+1}: {tool_name} with args: {tool_args}")
                
                # Find and execute the tool
                tool_result = None
                for tool_func in tools:
                    if tool_func.name == tool_name:
                        tool_result = tool_func.invoke(tool_args)
                        break
                
                if tool_result is None:
                    tool_result = f"Tool {tool_name} not found"
                    print(f"[ERROR] Tool not found: {tool_name}")
                
                print(f"[DEBUG] Tool result: {tool_result}")
                
                # Add tool result as a ToolMessage
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    )
                )
            
            # Use llm_with_tools for the final response
            print("[DEBUG] Getting final response from LLM...")
            final_response = llm_with_tools.invoke(messages)
            print(f"[DEBUG] Final response content: {final_response.content}")
            return {"result": final_response.content}
        else:
            # No tools needed, return direct response
            print(f"[DEBUG] No tools used. Direct response: {response.content}")
            return {"result": response.content}
            
    except Exception as e:
        print(f"[ERROR] Exception in invoke_agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"result": f"Error: {str(e)}"}

if __name__ == "__main__":
    app.run()
```

## Step 2: Create Requirements Files

Create `agent/requirements.txt`:

```
bedrock-agentcore>=0.1.0
bedrock-agentcore-starter-toolkit>=0.1.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-aws>=0.2.0
boto3>=1.34.0
```

Create `client/requirements.txt`:

```
boto3>=1.34.0
```

## Step 3: Test Locally (Optional)

Navigate to the agent directory and install dependencies:

```bash
cd agent
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start your agent locally:

```bash
python langchain_agent.py
```

In another terminal, test it:

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the current timestamp?"}'
```

**If you get an error about port 8080 being in use**, kill the process and try again:

```bash
kill -9 $(lsof -ti:8080)
python langchain_agent.py
```

Press `Ctrl+C` to stop the local agent when done testing.

## Step 4: Configure and Deploy

Configure your agent for deployment:

```bash
agentcore configure -e langchain_agent.py
```

During configuration:
- **Execution Role**: Press Enter to auto-create
- **ECR Repository**: Press Enter to auto-create
- **Dependency file**: Press Enter to use detected `requirements.txt`
- **Authorization**: Choose `no` for OAuth (uses your local AWS credentials/role)

Deploy to AWS:

```bash
agentcore launch
```

After successful deployment, you'll receive an **Agent ARN** like:

```
arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/langchain-agent-abc123
```

**Save this ARN** - you'll need it for testing and invoking the agent.

## Step 5: Test Your Deployed Agent

Test with the starter toolkit:

```bash
agentcore invoke '{"prompt": "What is the current timestamp?"}'
agentcore invoke '{"prompt": "Generate a random number between 1 and 1000"}'
agentcore invoke '{"prompt": "What is the SHA256 hash of hello?"}'
```

## Step 6: Invoke Programmatically with Boto3

Navigate to the client directory:

```bash
cd ../client
pip install -r requirements.txt
```

Create `client/test_client.py`:

```python
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
```

**Before running**, update the following in `test_client.py`:
- Replace `agent_arn` with your actual Agent ARN
- Replace `us-east-1` with your AWS region if different

Run the test:

```bash
python test_client.py
```

## Step 7: Find Your Resources

After deployment, view your resources in the AWS Console:

**Resource** | **Location**
--- | ---
Agent Logs | CloudWatch → Log groups → `/aws/bedrock-agentcore/runtimes/{agent-id}-DEFAULT`
Container Images | ECR → Repositories → `bedrock-agentcore-langchain-agent`
Build Logs | CodeBuild → Build history
IAM Role | IAM → Roles → Search for "BedrockAgentCore"

## Step 8: Clean Up

When you're done, delete the AWS resources:

```bash
agentcore destroy
```

This will remove:
- The AgentCore Runtime
- The ECR repository
- The IAM execution role (if auto-created)

## Expected Output

When you run the test client, you should see output like:

```
================================================================================
Testing LangChain Agent with Tools on Amazon Bedrock AgentCore Runtime
================================================================================

These tools prove the agent is actually executing code:
- Current timestamp (LLM training data is frozen)
- Random numbers (LLM cannot generate true randomness)
- UUIDs (LLM cannot create valid UUIDs)
- Cryptographic hashes (LLM cannot compute actual hashes)
- Date calculations (LLM cannot accurately compute date arithmetic)
================================================================================

================================================================================
Test 1/7: What is the current timestamp?
================================================================================
Result: The current timestamp is **2025-11-26T05:40:05.317229** (in ISO format).

This represents November 26, 2025, at 5:40:05 AM (and 317229 microseconds) UTC.

================================================================================
Test 2/7: Generate a random number between 1 and 1000
================================================================================
Result: The random number generated between 1 and 1000 is **718**.

================================================================================
Test 3/7: Generate a UUID for me
================================================================================
Result: I've generated a UUID for you: **fd45926a-f5c5-4173-8159-4e561d71ae98**

================================================================================
Test 4/7: What is the SHA256 hash of the word 'hello'?
================================================================================
Result: The SHA256 hash of the word 'hello' is:

**2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824**

================================================================================
Test 5/7: Convert 1073741824 bytes to human readable format
================================================================================
Result: 1073741824 bytes equals **1.0 GB** (gigabyte).

This is exactly 1024³ bytes, which is a common size you might recognize as 1 gibibyte (GiB) in binary notation, or simply 1 GB in common usage.

================================================================================
Test 6/7: What day of the week was January 1, 2000?
================================================================================
Result: January 1, 2000 was a **Saturday**.

================================================================================
Test 7/7: How many days between 2020-01-01 and 2025-12-31?
================================================================================
Result: There are **2,191 days** between 2020-01-01 and 2025-12-31 (approximately 313 weeks or 6 years).

================================================================================
Testing Complete!
================================================================================

================================================================================
VERIFICATION TEST: Calling timestamp tool twice to prove it's dynamic
================================================================================

Attempt 1:
  Timestamp: The current timestamp is **2025-11-26T05:40:51.092212** (in ISO format).

This represents November 26, 2025, at 5:40:51 AM (and 092212 microseconds) UTC.
  Waiting 3 seconds...

Attempt 2:
  Timestamp: The current timestamp is **2025-11-26T05:41:00.289825** (in ISO format).

This represents November 26, 2025, at 5:41:00 AM (and 289825 microseconds) UTC.

================================================================================
If the timestamps are different, the tool is definitely being called!
================================================================================
```

## How It Works

1. **LangChain Agent**: Creates an agent with tool-calling capabilities
2. **Custom Tools**: Defines tools that an LLM cannot perform on its own:
   - Current timestamp (LLMs have fixed training data)
   - Random number generation (LLMs cannot generate true randomness)
   - UUID generation (LLMs cannot create valid UUIDs)
   - Cryptographic hashing (LLMs cannot compute actual hashes)
   - Date calculations (LLMs cannot accurately compute date arithmetic)
3. **BedrockAgentCoreApp**: Wraps the agent for deployment
4. **@app.entrypoint**: Marks the function as the agent's entry point
5. **AgentCore Runtime**: Hosts the agent as a managed service in AWS

## Common Issues

### Permission Denied
- Verify AWS credentials: `aws sts get-caller-identity`
- Ensure you have required IAM permissions

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html)
- [Bedrock AgentCore Starter Toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit)
