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
