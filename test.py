import os
from dotenv import load_dotenv
import pyaudio
import time
import numpy as np
import dashscope
from dashscope.audio.asr import TranslationRecognizerChat, TranscriptionResult, TranslationResult, TranslationRecognizerCallback
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback

# 加载环境变量
load_dotenv(dotenv_path='qwen3-235b-a22b.env')

# 设置阿里云API密钥sk-6d97b536ede84fbbac9663e02fa7fa8b
dashscope.api_key = os.environ.get("DASHSCOPE_SPEECH_SPEECH_API_KEY", os.environ["key"])

# 阿里云语音识别配置
ASR_MODEL = "gummy-chat-v1"  # 或者 "paraformer-realtime-v2"
ASR_SAMPLE_RATE = 16000

# 阿里云语音合成配置
TTS_MODEL = "cosyvoice-v1"
TTS_VOICE = "aixiaobei" # 使用已知音色测试
TTS_SAMPLE_RATE = 22050

# 全局变量
mic = None
stream = None
recognized_text = ""
recognition_completed = False

# 语音识别回调类
class ASRCallback(TranslationRecognizerCallback):
    def __init__(self):
        global recognition_completed
        recognition_completed = False
        
    def on_open(self) -> None:
        global mic, stream
        print("语音识别连接已打开")
        mic = pyaudio.PyAudio()
        stream = mic.open(
            format=pyaudio.paInt16, channels=1, rate=ASR_SAMPLE_RATE, input=True
        )

    def on_close(self) -> None:
        global mic, stream, recognition_completed
        print("语音识别连接已关闭")
        if stream:
            stream.stop_stream()
            stream.close()
        if mic:
            mic.terminate()
        stream = None
        mic = None
        recognition_completed = True

    def on_event(
        self,
        request_id,
        transcription_result: TranscriptionResult,
        translation_result: TranslationResult,
        usage,
    ) -> None:
        global recognized_text
        if transcription_result is not None:
            text = transcription_result.text
            if text.strip():  # 只打印非空文本
                print(f"识别结果: {text}")
                recognized_text = text

# 语音合成相关回调
class TTSCallback(ResultCallback):
    def __init__(self):
        self._player = None
        self._stream = None
        self.completed = False
        self.error_occurred = False
        print("TTSCallback 初始化")

    def on_open(self):
        print("TTS 连接已打开")
        try:
            self._player = pyaudio.PyAudio()
            self._stream = self._player.open(
                format=pyaudio.paInt16, channels=1, rate=TTS_SAMPLE_RATE, output=True
            )
            print("TTS 音频流已打开")
        except Exception as e:
            print(f"打开 TTS 音频流失败: {e}")
            self.error_occurred = True

    def on_data(self, data: bytes) -> None:
        if self._stream and not self.error_occurred:
            try:
                self._stream.write(data)
            except Exception as e:
                print(f"写入 TTS 音频数据失败: {e}")
                self.error_occurred = True

    def on_complete(self):
        print("TTS 任务完成 (on_complete)")
        self.completed = True

    def on_error(self, message: str):
        print(f"TTS 任务失败: {message}")
        self.error_occurred = True

    def on_close(self):
        print("TTS 连接已关闭")
        time.sleep(0.2) # 等待缓冲播放
        if self._stream:
            try: 
                self._stream.stop_stream()
                self._stream.close()
            except Exception: 
                pass # 忽略关闭流错误
            print("TTS 音频流已关闭")
        if self._player:
            try: 
                self._player.terminate()
            except Exception: 
                pass # 忽略终止播放器错误
            print("TTS PyAudio 已终止")

def test_tts():
    """测试阿里云语音合成功能"""
    print("\n=== 语音合成测试开始 ===")
    
    # 检查 API Key
    if not dashscope.api_key or not dashscope.api_key.startswith('sk-'):
         print("错误：无法进行TTS测试，无效的 API Key")
         return

    sample_text = "你好，这是一个语音合成功能的测试。Hello, this is a test for text to speech."
    print(f"待合成文本: {sample_text}")
    
    callback = TTSCallback()
    
    try:
        synthesizer = SpeechSynthesizer(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=callback
        )
        
        print("发起 TTS 请求...")
        synthesizer.streaming_call(sample_text)
        synthesizer.streaming_complete()
        print("TTS 请求已发送，等待播放完成...")
        
        # 等待播放完成或出错
        start_wait = time.time()
        timeout = 60 # 最长等待60秒
        while not callback.completed and not callback.error_occurred and (time.time() - start_wait < timeout):
            time.sleep(0.1)
            
        if callback.error_occurred:
            print("测试失败：语音合成或播放出错。")
        elif not callback.completed:
            print("测试失败：语音合成超时。")
        else:
            print("测试成功：语音合成并播放完毕。")
            
    except dashscope.common.error.AuthenticationError as e:
        print(f"测试失败：阿里云认证错误 - {e}")
        print("请检查您的 API Key 是否有效并具有语音合成权限。")
    except Exception as e:
        print(f"测试失败：发生意外错误 - {e}")
        
    print("=== 语音合成测试结束 ===")

def test_voice_recognition():
    """测试语音识别功能"""
    global recognized_text, recognition_completed
    
    print("=== 语音识别测试开始 ===")
    print("请对着麦克风说话，将测试30秒...")
    
    # 重置识别结果
    recognized_text = ""
    recognition_completed = False
    
    # 测试时间
    test_duration = 30  # 测试30秒
    start_time = time.time()
    
    # 跟踪当前活动的识别器
    current_recognizer = None
    
    def create_new_recognizer():
        """创建并启动新的识别器"""
        callback = ASRCallback()
        recognizer = TranslationRecognizerChat(
            model=ASR_MODEL,
            format="pcm",
            sample_rate=ASR_SAMPLE_RATE,
            transcription_enabled=True,
            translation_enabled=False,
            callback=callback,
        )
        recognizer.start()
        return recognizer
    
    # 创建第一个识别器
    current_recognizer = create_new_recognizer()
    
    try:
        # 循环读取音频并发送
        while time.time() - start_time < test_duration:
            if stream:
                try:
                    # 读取音频数据
                    data = stream.read(3200, exception_on_overflow=False)
                    
                    # 发送音频帧
                    if not current_recognizer.send_audio_frame(data):
                        print("===== 句子结束，重新开始新的识别 =====")
                        
                        # 停止当前识别器
                        try:
                            current_recognizer.stop()
                        except:
                            pass
                            
                        # 创建新的识别器
                        current_recognizer = create_new_recognizer()
                        
                        # 暂停一小段时间等待新识别器准备好
                        time.sleep(0.3)
                    
                    # 简单的静音检测
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    if np.abs(audio_data).mean() < 100:  # 很低的音量阈值
                        # 静音状态，不做特殊处理
                        pass
                    
                    # 短暂暂停，减少CPU使用
                    time.sleep(0.01)
                except Exception as e:
                    print(f"处理音频时出错: {e}")
                    time.sleep(0.5)
            else:
                print("麦克风流不可用")
                break
    except KeyboardInterrupt:
        print("测试被用户中断")
    finally:
        # 停止识别
        print("停止识别...")
        if current_recognizer:
            try:
                current_recognizer.stop()
            except:
                pass
        
        # 等待识别完成
        timeout = 5  # 最多等待5秒
        wait_start = time.time()
        while not recognition_completed and time.time() - wait_start < timeout:
            time.sleep(0.1)
            
        print("=== 语音识别测试结束 ===")
        print(f"最终识别结果: {recognized_text}")

if __name__ == "__main__":
    # 选择要运行的测试
    # test_voice_recognition() 
    test_tts()