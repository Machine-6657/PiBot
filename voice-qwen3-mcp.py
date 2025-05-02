
import os  
from dotenv import load_dotenv  
import io  
import azure.cognitiveservices.speech as speechsdk  
from openai import OpenAI
import time  
import datetime  
import threading  
import json, ast  
#from tools import *
import requests  
from io import BytesIO  
import tempfile  
import numpy as np 
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
  
load_dotenv("qwen3-235b-a22b.env")  
client = OpenAI(api_key=os.environ["key"], base_url=os.environ["base_url"]) 
Azure_speech_key = os.environ["Azure_speech_key"]  
Azure_speech_region = os.environ["Azure_speech_region"]  
Azure_speech_speaker = os.environ["Azure_speech_speaker"]  
WakeupWord = os.environ["WakeupWord"]  
WakeupModelFile = os.environ["WakeupModelFile"]  

messages = []  

# Set up Azure Speech-to-Text and Text-to-Speech credentials  
speech_key = Azure_speech_key  
service_region = Azure_speech_region  
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)  
# Set up Azure Text-to-Speech language  
speech_config.speech_synthesis_language = "zh-CN"  
# Set up Azure Speech-to-Text language recognition  
speech_config.speech_recognition_language = "zh-CN"  
lang = "zh-CN"  
# Set up the voice configuration  
speech_config.speech_synthesis_voice_name = Azure_speech_speaker  
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)  
connection = speechsdk.Connection.from_speech_synthesizer(speech_synthesizer)  
connection.open(True)  
# Creates an instance of a keyword recognition model. Update this to  
# point to the location of your keyword recognition model.  
model = speechsdk.KeywordRecognitionModel(WakeupModelFile)  
# The phrase your keyword recognition model triggers on.  
keyword = WakeupWord  
# Set up the audio configuration  
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)  
# Create a speech recognizer and start the recognition  
#speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)  
auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["ja-JP", "zh-CN"])  
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config,  
                                               auto_detect_source_language_config=auto_detect_source_language_config)  
unknownCount = 0  
sysmesg = {"role": "system", "content": os.environ["sysprompt_zh-CN"]}  
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]


isListenning=False

llm_cfg = {
    # Use the model service provided by DashScope:
    'model': 'qwen3-235b-a22b',
    'model_server': 'dashscope',
    'api_key': os.environ["key"],
    'generate_cfg': {
        'top_p': 0.8,
        'thought_in_content': False,
    }
}
system_instruction = '''你是一个AI助理。/no think'''
with open("mcp_server_config.json", "r") as f:
    config = json.load(f)
    print(config)  


tools = [config] # `code_interpreter` is a built-in tool for executing code.
print(tools)
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools)


def display_text(s):
    print(s)
def speech_to_text():  
    global unknownCount  
    global lang,isListenning  
    print("Please say...")  
    result = speech_recognizer.recognize_once_async().get()  
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:  
        unknownCount = 0  
        isListenning=False
        return result.text  
    elif result.reason == speechsdk.ResultReason.NoMatch:  
        isListenning=False
        unknownCount += 1  
        error = os.environ["sorry_" + lang]  
        text_to_speech(error)  
        return '...'  
    elif result.reason == speechsdk.ResultReason.Canceled:  
        isListenning=False
        return "speech recognizer canceled." 
    

def getVoiceSpeed():  
    return 17  
  
def text_to_speech(text, _lang=None):  
    global lang  
    try:  
        result = buildSpeech(text).get()  
        #result = speech_synthesizer.speak_ssml_async(ssml_text).get()  
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:  
            print("Text-to-speech conversion successful.")  
            return "Done."  
        else:  
            print(f"Error synthesizing audio: {result}")  
            return "Failed."  
    except Exception as ex:  
        print(f"Error synthesizing audio: {ex}")  
        return "Error occured!"  
        
def buildSpeech(text, _lang=None):
    voice_lang = lang  
    voice_name = "zh-CN-XiaoxiaoMultilingualNeural"  
    ssml_text = f'''  
        <speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="{lang}"><voice name="{voice_name}"><lang xml:lang="{voice_lang}"><prosody rate="{getVoiceSpeed()}%">{text.replace('*', ' ').replace('#', ' ')}</prosody></lang></voice></speak>  
    '''  
    print(f"{voice_name} {voice_lang}!")  
    return speech_synthesizer.speak_ssml_async(ssml_text)
    
def getPlayerStatus():
    return bot._call_tool("music_player-isPlaying","{}")
def pauseplay():
    return bot._call_tool("music_player-pauseplay","{}")
    
def generate_text(prompt):  
    global messages  
    messages.append({"role": "user", "content": prompt})  
    #tools = getTools()  
    collected_messages = []
    last_tts_request = None

    split=True
    result=""
    index=0
    for response in bot.run(messages=messages):
        result = sentence = response[0]["content"]
        if sentence and len(sentence)>0 and sentence[-1] in tts_sentence_end and split: # 发现分段标记：句子结束
            sentence=sentence[index:]
            if len(sentence)>500 : #如果这句足够长，后面不再分段朗读了
                split=False
            elif len(sentence)<6: #如果这句太短了，不单独朗读，并入后面
                continue
            
            
            if sentence != '': # 如果句子里只有 \n or space则跳过朗读
                print(f"Speech synthesized to speaker for: {sentence}")
                index=len(result)
                #text=text.replace("<think>","让我先来思考一下：")
                #text=text.replace("</think>","嗯，我想好了，下面是我的回答。")
                last_tts_request = buildSpeech(sentence)
                
    # Append the bot responses to the chat history.
    messages.extend(response)
    if last_tts_request:
        last_tts_request.get() 
    return result
  
def Get_Chat_Deployment():  
    deploymentModel = os.environ["Azure_OPENAI_Chat_API_Deployment"]  
    return deploymentModel  
  

def recognized_cb(evt):  
    result = evt.result  
    if result.reason == speechsdk.ResultReason.RecognizedKeyword:  
        print("RECOGNIZED KEYWORD: {}".format(result.text))  
    global done  
    done = True  

def canceled_cb(evt):  
    result = evt.result  
    if result.reason == speechsdk.ResultReason.Canceled:  
        print('CANCELED: {}'.format(result.cancellation_details.reason))  
    global done  
    done = True  
   
def start_recognition():  
    global unknownCount ,isListenning ,playing
    while True:  
        keyword_recognizer = speechsdk.KeywordRecognizer()  
        keyword_recognizer.recognized.connect(recognized_cb)  
        keyword_recognizer.canceled.connect(canceled_cb) 
        first=os.environ["welcome_" + lang]
        display_text(first) 
        if getPlayerStatus()!='playing':
            text_to_speech(first)  
        isListenning=True
        result_future = keyword_recognizer.recognize_once_async(model)  
        while True:  
            result = result_future.get()
            # Read result audio (incl. the keyword).
            if result.reason == speechsdk.ResultReason.RecognizedKeyword:
                print("Keyword recognized")
                isListenning=False
                if getPlayerStatus()=='playing':
                    pauseplay() #被唤醒后，如果有音乐播放则暂停播放
                break
            time.sleep(0.1)  
            
        display_text("很高兴为您服务，我在听请讲。")  
        text_to_speech("很高兴为您服务，我在听请讲。")  
        
          
        while unknownCount < 2:
            isListenning=True
            user_input = speech_to_text()
            if user_input=='...':
                continue
            print(f"You: {user_input}")  
            
            display_text(f"You: {user_input}")  
            response = generate_text(user_input)
            print(getPlayerStatus())
            if getPlayerStatus()=='playing':
                break
            
            #text_to_speech(f"你说的是：{user_input}")  
            
          
        bye_text = os.environ["bye_" + lang]  
        display_text(bye_text) 
        if getPlayerStatus()!='playing':
            text_to_speech(bye_text)  
        
        unknownCount = 0  
        time.sleep(0.1)  

if __name__ == "__main__":
    start_recognition()