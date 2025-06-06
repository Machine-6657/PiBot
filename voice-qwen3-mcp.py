import os  
import signal  # 用于处理Ctrl+C
import sys
from dotenv import load_dotenv  
import io  
from openai import OpenAI
import re  
import time  
import datetime  
import threading  
import json, ast  
import requests  
from io import BytesIO  
import tempfile  
import numpy as np 
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
# 添加阿里云语音合成和识别库和PyAudio
import pyaudio
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback
# 替换为流式识别
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

load_dotenv(dotenv_path='qwen3-235b-a22b.env')
# 设置阿里云API密钥
# 优先从 DASHSCOPE_SPEECH_API_KEY 读取语音服务的Key，否则使用提供的Key
dashscope.api_key = os.environ.get("DASHSCOPE_SPEECH_API_KEY", "sk-6d97b536ede84fbbac9663e02fa7fa8b")
# 检查语音服务Key是否有效
if not dashscope.api_key or not dashscope.api_key.startswith('sk-'):
    print(f"错误：无效的语音服务API密钥: {dashscope.api_key}，请检查环境变量 DASHSCOPE_SPEECH_API_KEY 或直接提供的密钥。")
    sys.exit(1)

# 大模型的Key (从环境变量 key 读取)
llm_api_key = os.environ.get("key")
if not llm_api_key:
    print("错误：未找到大模型 API 密钥的环境变量 key。")
    sys.exit(1)

# 阿里云语音合成配置
TTS_MODEL = "cosyvoice-v1"
TTS_VOICE = "longwan" # 尝试兼容旧配置或用默认值
TTS_SAMPLE_RATE = 22050

# 阿里云语音识别配置 (使用流式模型)
ASR_MODEL = "paraformer-realtime-v2"  # 流式识别模型
ASR_SAMPLE_RATE = 16000

# 从环境变量中读取逗号分隔的唤醒词，提供默认值
WakeupWords_str = os.environ.get("WakeupWords") 
wake_word_list = [word.strip().lower() for word in WakeupWords_str.split(',') if word.strip()] # 分割成列表并转小写, 过滤空字符串
if not wake_word_list:
    print("警告：未配置有效的唤醒词，将使用默认 '小川同学'")
    wake_word_list = ["小川同学"]

messages = []  

# 麦克风和音频流的全局变量
mic = None
stream = None
recognized_text_buffer = "" # 用于累积识别文本
last_recognized_sentence = "" # 存储上一句完整识别结果
is_tts_speaking = False # 全局标志：TTS是否正在说话
stop_recognition_flag = threading.Event() # 用于优雅停止识别线程
asr_error_occurred = False # 标记ASR是否出错

# 创建TTS回调处理类
class TTSCallback(ResultCallback):
    def __init__(self):
        self._player = None
        self._stream = None
        self.completed = False
        self.error_occurred = False

    def on_open(self):
        global is_tts_speaking
        print("语音合成连接已打开")
        is_tts_speaking = True # TTS开始说话
        try:
            self._player = pyaudio.PyAudio()
            self._stream = self._player.open(
                format=pyaudio.paInt16, channels=1, rate=TTS_SAMPLE_RATE, output=True
            )
            print("TTS音频输出流已打开")
        except Exception as e:
            print(f"打开音频输出流失败: {e}")
            self.error_occurred = True
            is_tts_speaking = False # 出错时确保重置

    def on_data(self, data: bytes) -> None:
        if self._stream and not self.error_occurred:
            try:
                self._stream.write(data)
            except Exception as e:
                print(f"写入音频数据失败: {e}")
                self.error_occurred = True
                is_tts_speaking = False # 出错时确保重置
                # 尝试关闭流，避免持续错误
                try: 
                    self._stream.stop_stream(); 
                    self._stream.close(); 
                except: 
                    pass
                try: 
                    self._player.terminate(); 
                except: pass

    def on_complete(self):
        print("语音合成任务完成")
        self.completed = True
        # is_tts_speaking 将在 on_close 中设置

    def on_error(self, message: str):
        global is_tts_speaking
        print(f"语音合成任务失败: {message}")
        self.error_occurred = True
        is_tts_speaking = False # 出错时确保重置

    def on_close(self):
        global is_tts_speaking
        print("语音合成连接已关闭")
        # 确保播放完所有缓冲数据
        time.sleep(0.2) # 短暂等待
        if self._stream and not self.error_occurred:
            try: 
                self._stream.stop_stream(); 
                self._stream.close(); 
            except Exception as e: 
                print(f"关闭TTS音频流时出错: {e}")
        if self._player and not self.error_occurred:
            try: 
                self._player.terminate(); 
            except Exception as e: 
                print(f"终止TTS PyAudio时出错: {e}")
        is_tts_speaking = False # TTS结束说话

# 语音识别回调类 (使用RecognitionCallback)
class ASRCallback(RecognitionCallback):
    def __init__(self, listener_type="wake_word"):
        self.listener_type = listener_type # "wake_word" or "command"
        global recognized_text_buffer, last_recognized_sentence, asr_error_occurred
        recognized_text_buffer = "" # 清空缓冲区
        last_recognized_sentence = ""
        asr_error_occurred = False # 重置错误标志
        print(f"初始化 ASRCallback (类型: {self.listener_type})")

    def on_open(self) -> None:
        global mic, stream, asr_error_occurred
        print("语音识别连接已打开")
        try:
            # 确保之前的资源已释放
            if stream: stream.close()
            if mic: mic.terminate()
            mic = pyaudio.PyAudio()
            stream = mic.open(
                format=pyaudio.paInt16, channels=1, rate=ASR_SAMPLE_RATE, input=True, frames_per_buffer=3200 # 使用固定buffer
            )
            print("麦克风流已打开")
            asr_error_occurred = False
        except Exception as e:
            print(f"打开麦克风失败: {e}")
            asr_error_occurred = True
            stop_recognition_flag.set() # 通知主线程停止

    def on_close(self) -> None:
        print("语音识别连接已关闭")
        # 资源关闭移到主线程的 run_asr_listener finally 块

    def on_complete(self) -> None:
        print(f'语音识别完成 (类型: {self.listener_type})')

    def on_error(self, message) -> None:
        global asr_error_occurred
        print(f'语音识别错误 (类型: {self.listener_type}): {message.message}')
        asr_error_occurred = True
        stop_recognition_flag.set() # 通知主线程停止

    def on_event(self, result: RecognitionResult) -> None:
        global recognized_text_buffer, last_recognized_sentence, is_tts_speaking, asr_error_occurred

        if asr_error_occurred: return # 如果出错，不再处理

        if is_tts_speaking and self.listener_type == "wake_word":
            # print("[忽略ASR: TTSing]") # 调试
            return

        sentence = result.get_sentence()
        if sentence and 'text' in sentence:
            current_segment = sentence['text']
            if current_segment: # 仅在有文本时更新
                recognized_text_buffer = current_segment # 保存当前识别片段
                # print(f"实时 ({self.listener_type}): {recognized_text_buffer}") # 调试
                if RecognitionResult.is_sentence_end(sentence):
                    sentence_text = recognized_text_buffer.strip() # 获取当前完整句子文本
                    print(f"完整句 ({self.listener_type}): {sentence_text}")
                    if sentence_text: # 确保句子不为空
                        # 修改：追加句子而不是覆盖
                        if last_recognized_sentence: # 如果已有内容，加个空格分隔
                            last_recognized_sentence += " " + sentence_text
                        else:
                            last_recognized_sentence = sentence_text # 第一个句子直接赋值
                        print(f"累积指令: '{last_recognized_sentence}'") # 调试
                    recognized_text_buffer = "" # 句子结束，清空缓冲区

# 语言设置
lang = "zh-CN"  

# LLM Agent 初始化
llm_cfg = {
    'model': os.environ.get("model"), # 允许通过环境变量配置模型
    'model_server': 'dashscope',
    'api_key': llm_api_key, # 使用单独的大模型Key
    'generate_cfg': {
        'top_p': 0.8,
        'thought_in_content': False,#关闭思考模式？
    }
}
# 修改系统指令，要求输出纯文本，并要求总结工具结果，特别是多个工具的结果
# 恢复正确的 os.environ.get 用法：第一个参数是环境变量名，第二个是默认值
system_instruction = '''你是一个AI陪伴机器人，可以陪你聊天，安抚你的情绪，名字叫小川。
请在你的回答中避免使用任何Markdown格式，例如星号(**)、井号(#)、列表标记(-, 1.)、反引号(``)、(「」)、（【】）等，直接输出一段连贯、简洁的不超过200个字的纯文本方便语音合成。
当你调用工具（例如查询天气、获取时间、播放音乐等）后：
1. 不要直接返回工具的原始输出（比如 JSON 格式的数据）。
2. 如果一次用户请求触发了多次工具调用，你需要将所有工具返回的信息整合起来。
3. 根据整合后的信息，用一段自然流畅、连贯、简洁的中文来回答用户的完整问题，而不是简单地拼接各个工具的结果，融合成一个单一的、流畅的文本段落，并且明确禁止在任何情况下使用列表或项目符号。
例如，如果用户问"现在几点，天气怎么样？"，你应该回答类似"现在是下午2点30分，今天天气晴朗。"这样的完整句子，而不是返回时间和天气两个独立的信息块。
4.若需要获取热点新闻，可以去新浪网查看，但不要直接返回新浪网的网页内容，而是总结出热点新闻的标题和内容，用一段自然流畅、连贯、简洁的中文来回答用户的完整问题，而不是简单地拼接各个工具的结果，融合成一个单一的、流畅的文本段落，并且明确禁止在任何情况下使用列表或项目符号，每一类新闻一行。
5.当用户指令包含播放音乐的意图时，你必须调用相应的播放音乐工具。当用户指令包含“关闭播放”、“停止播放”或类似意图时，你必须调用名为 music_player-stopplay 的工具来实际停止音乐，而不是仅仅口头回复说音乐已停止。'''
tools = []
try:
    # Ensure correct indentation for the try block
    with open("mcp_server_config.json", "r", encoding='utf-8') as f: # Specify encoding
        config = json.load(f)
        # Ensure the loaded config is treated as a list if it's a single tool dict
        if isinstance(config, dict):
            tools = [config]
        elif isinstance(config, list):
            tools = config # If the file already contains a list of tools
        print("MCP工具配置已加载")
except FileNotFoundError:
    print("未找到 mcp_server_config.json，将不加载工具")
except json.JSONDecodeError:
    print("mcp_server_config.json 格式错误，将不加载工具")
except Exception as e: # Catch generic exceptions during tool loading
    print(f"加载工具配置时发生错误: {e}")

bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools)

def display_text(s):
    print(s)
  
# def clean_markdown(text):
#     """移除常见的Markdown格式，使TTS输出更干净。"""
#     # ... (保持原样)
#     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **粗体**
#     text = re.sub(r'__(.*?)__', r'\1', text)    # __粗体__
#     text = re.sub(r'\*(.*?)\*', r'\1', text)      # *斜体*
#     text = re.sub(r'_(.*?)_', r'\1', text)      # _斜体_
#     text = re.sub(r'`(.*?)`', r'\1', text)
#     text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE) # 无序列表
#     text = re.sub(r'^\s*(\d+\.|[a-zA-Z]\.)\s+', '', text, flags=re.MULTILINE) # 有序列表 (数字或字母)
#     text = re.sub(r'^\s*([-*_]){3,}\s*$', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
#     text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
#     text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'[ \t]{2,}', ' ', text)
#     text = re.sub(r'\n{3,}', '\n\n', text) 
#     text = text.strip()
#     # 移除可能残留的列表标记行
#     text = re.sub(r'^\s*(\d+\.|[a-zA-Z]\.)\s*$', '', text, flags=re.MULTILINE)
#     text = re.sub(r'\n{2,}', '\n', text).strip()
#     return text

def text_to_speech(text, _lang=None):  
    """使用阿里云语音合成服务将文本转换为语音，并管理 is_tts_speaking 标志"""
    global is_tts_speaking
    if is_tts_speaking:
        print("警告：TTS 已经在说话，忽略新的请求")
        return "忙碌中。"

    try:
        cleaned_text = text
        if not cleaned_text:
            print("TTS: 文本为空，跳过合成")
            return "完成。"

        print(f"TTS: 准备合成: {cleaned_text}")
        is_tts_speaking = True # TTS开始
        callback = TTSCallback()

        synthesizer = SpeechSynthesizer(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=callback
        )

        # 直接调用 streaming_call 和 streaming_complete
        # DashScope SDK 会处理异步播放
        synthesizer.streaming_call(cleaned_text)
        synthesizer.streaming_complete() # 通知服务器文本已发送完毕

        # 不需要手动创建和管理线程了
        # tts_thread = threading.Thread(target=synthesizer.streaming_call, args=(cleaned_text,))
        # tts_thread.start()
        # synthesizer.streaming_complete() # 不应在这里立即调用

        print("TTS: 请求已发送，将在后台播放")
        # 主循环通过检查 is_tts_speaking == False 来判断播放是否结束
        # 或者可以添加一个阻塞等待，如果需要确保播放完再继续
        # while not callback.completed and not callback.error_occurred:
        #     time.sleep(0.1)
        # if callback.error_occurred: return "发生错误！"
        # else: return "完成。"
        # 当前设计为非阻塞
        return "处理中。"

    except Exception as ex:  
        print(f"TTS: 意外错误: {ex}")
        is_tts_speaking = False # 确保异常时重置标志
        # 尝试清理可能残留的播放器资源
        if 'callback' in locals() and hasattr(callback, '_player') and callback._player:
            try: callback._player.terminate()
            except: pass
        return "发生错误！"
    
def getPlayerStatus():
    if is_tts_speaking: return "stopped" # TTS说话时，音乐应停止
    try:
        result = bot._call_tool("music_player-getPlaybackStatus", "{}")
        # 假设工具返回包含状态的JSON字符串或字典
        if isinstance(result, str):
            status_data = json.loads(result)
        elif isinstance(result, dict):
            status_data = result
        else:
            return "unknown" # 未知格式
        # 根据实际工具返回格式调整
        return status_data.get("status", "stopped")
    except Exception as e:
        print(f"获取播放器状态时出错: {e}")
        return "stopped" # 出错时默认为停止

def pauseplay():
    if is_tts_speaking: return "ignored" # TTS说话时不操作播放器
    try:
        return bot._call_tool("music_player-pauseplay","{}")
    except Exception as e:
        print(f"暂停/播放音乐时出错: {e}")
        return "error"
    
def unpauseplay():
    if is_tts_speaking: return "ignored" # TTS说话时不操作播放器
    try:
        return bot._call_tool("music_player-unpauseplay","{}")
    except Exception as e:
        print(f"恢复播放音乐时出错: {e}")
        return "error"

def generate_text(prompt):  
    """使用大模型生成文本，并调用TTS播放回复"""
    global messages, is_tts_speaking # Declare both globals at the top
    messages.append({"role": "user", "content": prompt})  

    full_response_text = ""
    print("AI: ", end='', flush=True) # 打印前缀

    # --- 添加：思考提示逻辑 ---
    start_time = time.time() # 记录开始时间
    thinking_threshold = 5.5 # 超过多少秒提示（可调整）
    thinking_message_played = False # 是否已播放提示
    # ------------------------

    try:
        # 使用bot.run生成文本
        response_generator = bot.run(messages=messages)
        for response in response_generator:
            # --- 添加：检查是否需要播放思考提示 ---
            elapsed_time = time.time() - start_time
            # 检查条件：超过阈值 & 未播放过 & TTS当前空闲
            if elapsed_time > thinking_threshold and not thinking_message_played and not is_tts_speaking:
                print("\n[AI思考中...] ", end='') # 控制台提示
                text_to_speech("稍等片刻，正在思考哦") 
                thinking_message_played = True
            # ---------------------------------

            # *** MODIFIED DEBUGGING PRINT ***
            #print(f"DEBUG voice-qwen3-mcp.py: generate_text - bot.run() response: {response}") 

            # --- 添加：如果正在播放思考提示，则等待 --- 
            # 避免思考提示和实际回复语音重叠
            # --- REMOVED BLOCK causing hang ---
            # if is_tts_speaking and thinking_message_played:
            #     continue # 跳过本次循环，等待TTS说完思考提示
            # ---------------------------------------

            # 检查响应格式是否符合预期，并确保我们处理的是 'assistant' 的最终回复
            if response and isinstance(response, list) and response[-1]:
                last_message = response[-1]
                if isinstance(last_message, dict) and last_message.get('role') == 'assistant' and "content" in last_message:
                    chunk_content = last_message["content"]
                    new_text = chunk_content[len(full_response_text):]
                    if new_text:
                        print(new_text, end='', flush=True) # 流式打印文本
                        full_response_text = chunk_content # 更新完整文本
            # else: # 注释掉，避免打印不必要的格式异常
                # 处理可能的错误或空响应
                # print(f"[LLM响应格式异常: {response}]")
                # break # 避免后续处理错误

        print() # 换行

        # --- 添加：如果播放了思考提示，等待其完成 --- 
        if thinking_message_played:
            print("[等待思考提示TTS完成...]")
            while is_tts_speaking:
                time.sleep(0.1)
            print("[思考提示TTS完成]")
        # ------------------------------------------

        # 将最终的完整回复添加到历史记录
        # Qwen-Agent 的 response list 包含了所有历史记录+当前回复
        # 我们需要找到最终的助手回复并更新 messages
        # --- 修改历史记录管理方式 --- 
        # if response and isinstance(response, list):
        #     messages = response # 直接用最后一次返回的完整列表更新历史
        # else:
        #      # 如果循环未产生有效response或出错，手动添加
        #      print("[警告] LLM未返回有效响应结构，手动记录文本。")
        #      # 只有在 full_response_text 非空时才添加，避免添加空消息
        #      if full_response_text:
        #         messages.append({"role": "assistant", "content": full_response_text})
        
        # 手动追加助手的最终回复到 history
        if full_response_text:
            messages.append({"role": "assistant", "content": full_response_text})
        elif not asr_error_occurred: # 如果ASR没出错，但LLM没回复，也可能需要记录点什么？（可选）
             print("[信息] LLM未生成回复内容。")
             # 可以选择不添加空消息，或者添加一个标记？目前选择不添加。
        # --------------------------

        # 调用TTS播放完整回复
        if full_response_text:
            text_to_speech(full_response_text)

        return full_response_text

    except Exception as e:
        print(f"\n生成文本或TTS时出错: {e}")
        # 确保TTS标志在出错时被重置
        is_tts_speaking = False
        error_message = f"抱歉，处理时遇到错误。" # 简化错误信息
        # 尝试记录错误到历史
        if isinstance(messages, list):
             messages.append({"role": "assistant", "content": error_message})
        return error_message

def run_asr_listener(listener_type="wake_word"):
    """运行一个语音识别监听器实例（唤醒词或指令）"""
    global stop_recognition_flag, mic, stream, recognized_text_buffer, last_recognized_sentence, asr_error_occurred

    print(f"启动 {listener_type} 监听器...")
    callback = ASRCallback(listener_type)
    recognition = None # 初始化

    try:
        recognition = Recognition(
            model=ASR_MODEL,
            format='pcm',
            sample_rate=ASR_SAMPLE_RATE,
            semantic_punctuation_enabled=False, # 标点可能干扰唤醒词
            callback=callback)

        recognition.start() # start 可能抛出异常，特别是网络或认证问题
        stop_recognition_flag.clear() # 重置停止标志
        recognized_text_buffer = "" # 清空缓冲区
        last_recognized_sentence = ""
        asr_error_occurred = False # 重置错误标志

        silent_frames = 0
        max_silent_time = 5 # 指令模式下5秒静音则停止 (原为3秒)
        frame_size = 3200 # 与mic open一致
        max_silent_frames = int(max_silent_time * ASR_SAMPLE_RATE / frame_size)
        start_listen_time = time.time()
        max_command_listen_time = 20 # 指令模式最长听15秒

        while not stop_recognition_flag.is_set():
            if asr_error_occurred: # 如果回调报告错误，退出
                 print(f"{listener_type} 监听器：检测到错误，退出。")
                 break

            if stream and not stream.is_stopped(): # 检查流是否有效
                try:
                    data = stream.read(frame_size, exception_on_overflow=False)
                    if stop_recognition_flag.is_set(): break # 再次检查停止标志
                    recognition.send_audio_frame(data)

                    # 指令模式下的静音和超时检测
                    if listener_type == "command":
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        if np.abs(audio_data).mean() < 100: # 简单的能量检测
                            silent_frames += 1
                            if silent_frames > max_silent_frames:
                                print("指令监听：检测到静音，停止。")
                                stop_recognition_flag.set() # 发送停止信号
                        else:
                            silent_frames = 0
                        if time.time() - start_listen_time > max_command_listen_time:
                            print("指令监听：超时，停止。")
                            stop_recognition_flag.set()

                except IOError as e:
                    print(f"音频流读取错误: {e}, 停止监听")
                    asr_error_occurred = True
                    stop_recognition_flag.set()
                    break
                except AttributeError as e:
                     # 捕获 stream 可能为 None 的情况
                     print(f"音频流属性错误（可能已关闭）: {e}")
                     asr_error_occurred = True
                     stop_recognition_flag.set()
                     break
                except Exception as e:
                    print(f"发送音频帧时未知错误: {e}")
                    asr_error_occurred = True
                    stop_recognition_flag.set()
                    break
            else:
                print(f"{listener_type} 监听器：音频流不可用或已停止，退出。")
                asr_error_occurred = True # 标记错误状态
                stop_recognition_flag.set()
                break # 退出循环

            time.sleep(0.01) # 避免CPU过度占用

    except dashscope.common.error.AuthenticationError as e:
        print(f"阿里云认证错误: {e}")
        asr_error_occurred = True
        stop_recognition_flag.set()
    except Exception as e:
        print(f"启动或运行 {listener_type} 监听器时出错: {e}")
        asr_error_occurred = True
        stop_recognition_flag.set() # 确保出错时也尝试停止
    finally:
        print(f"--- 清理 {listener_type} 监听器资源 ---")
        if recognition:
            try:
                print("尝试停止识别器...")
                recognition.stop()
                print("识别器已停止")
            except Exception as e:
                print(f"停止识别器 {listener_type} 时出错: {e}")

        # 在这里关闭流和mic，因为回调的on_close可能不总被调用或有时序问题
        if stream:
            try:
                stream.stop_stream()
                stream.close()
                print("音频流已关闭")
            except Exception:
                pass # 忽略关闭流时的错误
            stream = None # 明确置为None
        if mic:
            try:
                mic.terminate()
            except Exception:
                pass # 忽略终止mic时的错误
            mic = None # 明确置为None
        print(f"--- {listener_type} 监听器资源清理完毕 ---")
   
def start_recognition():
    global recognized_text_buffer, last_recognized_sentence, is_tts_speaking, asr_error_occurred

    print(f"当前工作目录: {os.getcwd()}")
    print(f"使用的唤醒词列表: {wake_word_list}") 

    # --- 启动时说欢迎语 ---
    first = os.environ.get("welcome_" + lang, "你好！我是小川。") # 默认欢迎语
    display_text(first)
    initial_status = getPlayerStatus()
    print(f"初始播放器状态: {initial_status}")
    if initial_status != 'playing':
        text_to_speech(first)
        # 稍微等待TTS播放完毕，避免ASR立即启动冲突
        time.sleep(0.5) # 稍微增加等待时间
        while is_tts_speaking: time.sleep(0.1) # 确保TTS完全结束

    # --- 主循环与状态管理 ---
    current_mode = "WAKE_WORD" # 初始状态为等待唤醒词
    music_was_playing_before_interaction = False # 新增状态变量，用于跟踪唤醒前音乐是否在播放
    
    while True:
        if current_mode == "WAKE_WORD":
            # --- 步骤1: 持续监听唤醒词 ---
            print("\n=== 开始监听唤醒词 ===")
            # music_was_playing_before_interaction = False # 在这里重置会导致唤醒失败时丢失状态，移到后面更合适
            asr_error_occurred = False # 重置错误标志
            last_recognized_sentence = "" # 重置句子
            listener_thread = threading.Thread(target=run_asr_listener, args=("wake_word",), daemon=True)
            listener_thread.start()

            wake_detected = False
            while listener_thread.is_alive():
                if asr_error_occurred: # 如果ASR线程内部出错，退出等待
                    print("唤醒词监听因错误停止。")
                    break

                current_sentence = last_recognized_sentence # 使用完整的句子检查
                if current_sentence:
                    print(f"检查唤醒词: '{current_sentence}'") # 调试
                    lower_text = current_sentence.lower()
                    if any(word in lower_text for word in wake_word_list):
                        print(f"检测到唤醒词: {current_sentence}")
                        wake_detected = True
                        stop_recognition_flag.set() # 发送停止信号
                        break # 退出唤醒词检测循环
                    else:
                        # 不是唤醒词，重置等待下一句完整识别
                        last_recognized_sentence = ""

                time.sleep(0.2) # 避免CPU空转

            # 等待监听线程完全停止
            print("等待唤醒词监听线程停止...")
            listener_thread.join(timeout=10) # 增加超时时间
            if listener_thread.is_alive():
                 print("警告：唤醒词监听线程未能按时停止。")
                 # 可能需要更强的停止机制

            if not wake_detected or asr_error_occurred:
                 print("未检测到唤醒词或发生错误，重新开始监听...")
                 # 如果因为唤醒失败或错误而重新监听，检查之前是否暂停了音乐
                 if music_was_playing_before_interaction:
                     print("唤醒失败或错误，但之前音乐已暂停，尝试恢复...")
                     unpauseplay()
                 music_was_playing_before_interaction = False # 无论如何都重置状态
                 time.sleep(2) # 暂停一下再重试
                 continue # 循环回到开始

            print("唤醒词监听已停止。")

            # --- 步骤2: 交互提示 (成功唤醒后) --- 
            current_status = getPlayerStatus()
            print(f"唤醒时播放器状态: {current_status}")
            if current_status == 'playing':
                print("音乐播放中，主动暂停...")
                pauseplay()
                music_was_playing_before_interaction = True # 记录音乐在唤醒前是播放状态
            else:
                music_was_playing_before_interaction = False # 确保如果音乐未播放，此标志为False
            
            text_to_speech("请讲。")
            while is_tts_speaking: time.sleep(0.1) # 等待"请讲"说完
            
            current_mode = "COMMAND" # 切换到指令模式
            # 直接进入下一次循环的COMMAND分支
            continue 

        elif current_mode == "COMMAND":
            # --- 步骤3: 监听用户指令 (或后续指令) --- 
            print("\n=== 开始监听指令 ===")
            recognized_text_buffer = "" # 清空缓冲区
            last_recognized_sentence = ""
            asr_error_occurred = False # 重置错误标志

            # 在主线程运行指令监听，简化状态管理
            run_asr_listener("command") 

            user_command = last_recognized_sentence # 获取最终结果
            print(f"最终识别指令: '{user_command}'")

            # --- 步骤4: 处理指令并回复 --- 
            if asr_error_occurred:
                 print("指令识别过程中发生错误。")
                 text_to_speech("抱歉，识别时遇到问题。")
                 while is_tts_speaking: time.sleep(0.1)
                 # 准备退下，检查是否需要恢复音乐
                 if music_was_playing_before_interaction:
                     print("因错误退出指令模式，尝试恢复之前播放的音乐...")
                     unpauseplay()
                     music_was_playing_before_interaction = False # 重置标志
                 current_mode = "WAKE_WORD" # 出错返回唤醒模式
            elif user_command: # 确保有指令且ASR未出错
                _full_response_text = generate_text(user_command) # 生成回复并用TTS播放
                while is_tts_speaking: time.sleep(0.1) # 等待回复播放完毕

                # 在指令处理完毕后，检查播放器的实际状态
                # 如果此时播放器已经是停止状态（比如用户指令是停止播放），
                # 那么就不应该在后续超时后自动恢复播放。
                try:
                    # 直接调用 music_player-getPlaybackStatus 工具获取最新状态
                    status_result_str = bot._call_tool("music_player-getPlaybackStatus", "{}") 
                    status_data = json.loads(status_result_str)
                    current_player_status_after_command = status_data.get("status", "unknown")
                    print(f"[DEBUG] Player status after user command processing: {current_player_status_after_command}")
                    if current_player_status_after_command == "stopped":
                        print("[DEBUG] Music player is confirmed stopped after command. Resetting 'music_was_playing_before_interaction'.")
                        music_was_playing_before_interaction = False
                except Exception as e:
                    print(f"[ERROR] Failed to get player status after command execution: {e}")

                print("--- 等待后续指令 (或静音超时) ---") 
                # music_was_playing_before_interaction 状态在此保持，因为交互可能继续 (除非上面重置了)
                current_mode = "COMMAND" 
            else: # 未识别到有效指令 (静音超时)
                print("未识别到有效指令或超时。")
                text_to_speech("我先退下了。")
                while is_tts_speaking: time.sleep(0.1)
                # 准备退下，检查是否需要恢复音乐
                if music_was_playing_before_interaction:
                    print("超时退出指令模式，尝试恢复之前播放的音乐...")
                    unpauseplay()
                    music_was_playing_before_interaction = False # 重置标志
                current_mode = "WAKE_WORD" # 超时返回唤醒模式

        # --- 安全间隔 --- 
        time.sleep(0.1) # 主循环间的短暂间隔

# 处理Ctrl+C信号
def signal_handler(sig, frame):
    print('\n程序被中断 (Ctrl+C)')
    global stop_recognition_flag
    stop_recognition_flag.set() # 通知所有监听线程停止
    print("正在停止监听线程...")
    # 给线程一点时间停止
    time.sleep(2)
    # 尝试清理 PyAudio 资源 (以防万一)
    global mic, stream
    print("清理音频资源...")
    if stream:
        try: 
            stream.stop_stream(); stream.close(); 
        except: pass
        stream = None
    if mic:
        try: 
            mic.terminate(); 
        except: pass
        mic = None
    print("资源已清理，程序退出。")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler) # 注册信号处理器
    start_recognition()