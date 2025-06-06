import os
import json
from dotenv import load_dotenv
from qwen_agent.agents import Assistant

# Load environment variables from qwen3-235b-a22b.env
load_dotenv(dotenv_path='qwen3-235b-a22b.env')

# Use the same API key setup as in the main script
# Speech key is not strictly needed here, but dashscope library might require a default key
# dashscope.api_key = os.environ.get("DASHSCOPE_SPEECH_API_KEY", "YOUR_FALLBACK_SPEECH_KEY") # Or remove if not needed

# Get LLM API key
llm_api_key = os.environ.get("key")
if not llm_api_key:
    print("错误：未找到大模型 API 密钥的环境变量 key。")
    exit(1)

# LLM Agent Initialization (Copied from voice-qwen3-mcp.py)
llm_cfg = {
    'model': os.environ.get("model", "qwen-plus"), # Use model from env or default to qwen-plus
    'model_server': 'dashscope',
    'api_key': llm_api_key,
    'generate_cfg': {
        'top_p': 0.8,
        'thought_in_content': False,
    }
}

# System Instruction (Copied from voice-qwen3-mcp.py)
# Note: Removed the repeated section and the final "不需要你深度思考" for clarity in testing
system_instruction = '''你是一个AI助理,名字叫小川。
请在你的回答中避免使用任何Markdown格式，例如星号(**)、井号(#)、列表标记(-, 1.)、反引号(``)等，直接输出一段连贯的纯文本方便语音合成。
当你调用工具（例如查询天气、获取时间、播放音乐等）后：
1. 不要直接返回工具的原始输出（比如 JSON 格式的数据）。
2. 如果一次用户请求触发了多次工具调用，你需要将所有工具返回的信息整合起来。
3. 根据整合后的信息，用一段自然流畅、连贯的中文来回答用户的完整问题，而不是简单地拼接各个工具的结果，融合成一个单一的、流畅的文本段落，并且明确禁止在任何情况下使用列表或项目符号。
例如，如果用户问"现在几点，天气怎么样？"，你应该回答类似"现在是下午2点30分，今天天气晴朗。"这样的完整句子，而不是返回时间和天气两个独立的信息块。'''

# Load Tools (Copied from voice-qwen3-mcp.py)
tools = []
tool_config_file = "mcp_server_config.json"
try:
    with open(tool_config_file, "r", encoding='utf-8') as f: # Specify encoding
        config = json.load(f)
        # Ensure the loaded config is treated as a list if it's a single tool dict
        if isinstance(config, dict):
             tools = [config]
        elif isinstance(config, list):
             tools = config # If the file already contains a list of tools
        print(f"MCP工具配置已从 {tool_config_file} 加载")
except FileNotFoundError:
    print(f"未找到 {tool_config_file}，将不加载工具")
except json.JSONDecodeError:
    print(f"{tool_config_file} 格式错误，将不加载工具")
except Exception as e:
    print(f"加载工具配置时发生错误: {e}")


# Initialize Assistant
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools)

# Conversation History
messages = []

print("----- Qwen Agent Test -----")
print("输入 '退出' 来结束对话。")

while True:
    try:
        # Get user input
        user_input = input("你: ")
        if user_input.lower() == '退出':
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Run the agent
        response_generator = bot.run(messages=messages)

        full_response_text = ""
        print("AI: ", end='', flush=True)

        # Process the streaming response
        final_response_list = None # Store the last yielded list
        for response in response_generator:
            final_response_list = response # Keep track of the latest full list
            # Check the last message in the list for assistant's content
            if response and isinstance(response, list) and response[-1]:
                 last_message = response[-1]
                 if isinstance(last_message, dict) and last_message.get('role') == 'assistant' and "content" in last_message:
                    chunk_content = last_message["content"]
                    # Calculate only the new part of the text
                    new_text = chunk_content[len(full_response_text):]
                    if new_text:
                        print(new_text, end='', flush=True) # Stream print
                        full_response_text = chunk_content # Update full text

        print() # Newline after AI response

        # Add assistant's final response to history for context
        # Use the extracted full_response_text, not the raw list from generator
        if full_response_text:
             messages.append({"role": "assistant", "content": full_response_text})
        elif final_response_list: # Fallback if only tool call/result happened
            # If the last message in the final list was a tool result, record it (might need adjustment)
            if final_response_list[-1].get('role') == 'tool':
                 print(f"[记录工具结果: {final_response_list[-1].get('content')}]")
                 messages.append(final_response_list[-1]) # Add tool message
            else:
                 print("[信息] AI 未生成文本回复。")
        else:
             print("[信息] AI 未生成文本回复。")


    except EOFError: # Handle Ctrl+D
        break
    except KeyboardInterrupt: # Handle Ctrl+C
        print("对话中断。")
        break
    except Exception as e:
        print(f"发生错误: {e}")
        # Optionally clear history on error or try to recover
        # messages = [] # Uncomment to clear history on error
        break # Exit on error for simplicity

print("----- 对话结束 -----")
