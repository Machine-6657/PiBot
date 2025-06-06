# mcp_server_onlinemusic_player.py
from mcp.server.fastmcp import FastMCP
from typing import Annotated
from pathlib import Path

from pydantic import BaseModel, Field
import json, ast

import requests, json
from io import BytesIO 
import tempfile 
import time
import datetime  
import io 
#import dateutil.parser  
import locale 
import os
import platform
#from dotenv import load_dotenv  
import subprocess  

# ---- 日志记录配置 ----
import logging
LOG_FILE = 'music_player_debug.log'
# 清空旧的日志文件
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except:
        pass # 忽略删除错误

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'), # 输出到文件
        # logging.StreamHandler() # 如果也想在控制台看到，可以取消注释，但stdio模式下可能干扰
    ]
)
logger = logging.getLogger(__name__)
logger.info("音乐播放器服务日志模块已初始化")
# ---- 日志记录配置结束 ----

# 设置 XDG_RUNTIME_DIR 环境变量 - 跨平台兼容
if platform.system() == 'Windows':
    # Windows上使用TEMP环境变量
    xdg_runtime_dir = os.environ.get('TEMP', 'C:\\\\Temp') # 确保反斜杠被正确转义
else:
    # Linux路径
    try:
        xdg_runtime_dir = '/run/user/{}'.format(os.getuid())
    except AttributeError:
        # 备用方案
        xdg_runtime_dir = '/tmp'

os.environ['XDG_RUNTIME_DIR'] = xdg_runtime_dir
import pygame   

quitReg=False
pause=False
playing=False
# Create an MCP server
mcp = FastMCP("music_player")
# 初始化pygame  
pygame.init()  
pygame.mixer.init() # 显式初始化 mixer

# 用于存储当前播放的临时文件名，方便清理
current_temp_file = None

@mcp.tool()
def search_music(song_name:str)-> str:
    """
    搜索音乐
    
    params: 
        song_name: 歌曲名或关键字
    
    """
    logger.info(f"收到搜索音乐请求: {song_name}")
    #return f"为您找到歌曲：{song_name} 已开始播放。如果有其他任务请告知我，我先退下了。"
    url='http://music.163.com/api/search/get/web?csrf_token=hlpretag=&hlposttag=&s= %s&type=1&offset=0&total=true&limit=10' % song_name
    try:
        res=requests.get(url, timeout=10)
        res.raise_for_status()
        logger.info(f"搜索API成功，返回文本长度: {len(res.text)}")
        # 这里应该直接返回res.text，后续的json.loads和songs拼接逻辑实际上不会执行，因为被return了
        # 如果打算处理后返回，那么return res.text需要移到后面或修改逻辑
        return res.text # MCP工具通常期望返回JSON字符串或可JSON序列化的Python对象
    except requests.exceptions.RequestException as req_e:
        logger.error(f"搜索歌曲API请求失败: {req_e}")
        # 对于工具来说，抛出异常或者返回一个表示错误的JSON可能更好
        # 例如: return json.dumps({"error": "网络请求失败，无法搜索歌曲。"})
        # 但当前MCP框架如何处理非JSON字符串返回或异常，需要根据其设计来定
        # 为了保持原样但记录日志，我们返回一个错误字符串，这可能导致调用方JSON解析失败
        return "Error: 网络请求失败，无法搜索歌曲。" # 这不是JSON，调用方可能会出问题
    # music_json=json.loads(res.text) # 这部分代码在 return res.text 之后，不会执行
    # count=music_json["result"]["songCount"]
    # songs=""
    # if count>0:
    #     for i in range(count-1):
    #         songs+=f'''[{music_json["result"]["songs"][i]["name"]}]'''
    # return songs
       
@mcp.tool()
def play_music(song_name:str)-> str:
    """
    播放音乐
    
    params: 
        song_name: 歌曲名或关键字
    
    """
    global playing, pause, current_temp_file
    logger.info(f"play_music 调用，歌曲名: {song_name}")

    if playing or pygame.mixer.music.get_busy(): # 检查pygame是否也认为在播放
        logger.debug("检测到已有音乐在播放或mixer仍繁忙，先停止并卸载旧的播放...")
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload() # 卸载音乐以释放文件
            logger.debug("旧音乐已停止并卸载")
            if current_temp_file and os.path.exists(current_temp_file):
                try:
                    os.remove(current_temp_file)
                    logger.debug(f"清理了旧的临时文件: {current_temp_file}")
                    current_temp_file = None
                except Exception as e:
                    logger.error(f"清理旧临时文件 {current_temp_file} 时出错: {e}")
        except Exception as e:
            logger.error(f"停止旧音乐、卸载或清理文件时出错: {e}")
    playing = False
    pause = False
    
    logger.debug("正在搜索歌曲...")
    url='http://music.163.com/api/search/get/web?csrf_token=hlpretag=&hlposttag=&s= %s&type=1&offset=0&total=true&limit=10' % song_name
    try:
        res=requests.get(url, timeout=10) 
        res.raise_for_status()
        music_json=json.loads(res.text)
        logger.info(f"歌曲搜索API成功，找到歌曲数量: {music_json.get('result', {}).get('songCount', 0)}")
    except requests.exceptions.RequestException as req_e:
        logger.error(f"搜索歌曲API请求失败: {req_e}")
        return json.dumps({"error": "网络请求失败，无法搜索歌曲。"}) # 返回JSON错误
    except json.JSONDecodeError as json_e:
        logger.error(f"解析歌曲搜索结果失败: {json_e}")
        return json.dumps({"error": "解析歌曲数据失败。"}) # 返回JSON错误
        
    count = music_json.get("result", {}).get("songCount", 0)
    
    if count > 0:
        musicName = downloadAndPlay(music_json, 0)
        if musicName:
            logger.info(f"找到歌曲：'{musicName}' 开始播放。请欣赏。")
            return json.dumps({"status":f"歌曲【{musicName}】已开始播放。"})
        else:
            playing=False
            pause = False
            logger.warning("downloadAndPlay 未成功播放音乐")
            return json.dumps({"error": "播放音乐失败，请检查网络或稍后再试。"})
    
    playing=False
    pause = False
    logger.info("没有找到符合条件的音乐")
    return json.dumps({"status": "没有找到音乐。"})

def downloadAndPlay(music_json,index):
    global playing, pause, current_temp_file
    logger.debug(f"downloadAndPlay 调用, index: {index}")
    
    if current_temp_file and os.path.exists(current_temp_file):
        try:
            os.remove(current_temp_file)
            logger.debug(f"清理了上一个临时文件: {current_temp_file}")
        except Exception as e:
            logger.error(f"清理旧临时文件时出错: {e}")
        current_temp_file = None

    count = music_json.get("result", {}).get("songCount", 0)
    if index >= count:
        logger.debug(f"downloadAndPlay: 索引 {index} 超出歌曲数量 {count}")
        playing = False 
        pause = False
        return False
        
    songs = music_json.get("result", {}).get("songs", [])
    if not songs or index >= len(songs):
        logger.warning(f"downloadAndPlay: 歌曲列表为空或索引无效")
        playing = False
        pause = False
        return False

    song_info = songs[index]
    songid = song_info.get("id")
    songName = song_info.get("name", "未知歌曲")
    
    if not songid:
        logger.error(f"歌曲信息中缺少ID: {song_info}")
        playing = False; pause = False
        return downloadAndPlay(music_json, index + 1)

    url='http://music.163.com/song/media/outer/url?id=%s.mp3' % songid
    logger.info(f"准备下载歌曲: {songName} (ID: {songid}), URL: {url}")
    
    temp_file = None
    temp_file_name_for_this_attempt = None
    response_obj = None # 重命名以避免与外部的response混淆，并初始化

    try:
        logger.debug(f"[{songName}] Calling requests.get(url, timeout=15)...") # 修复：直接记录将使用的超时值
        response_obj = requests.get(url, timeout=15) # timeout参数单位为秒
        logger.debug(f"[{songName}] requests.get() returned. Status: {response_obj.status_code if response_obj else 'No response_obj'}")

        logger.debug(f"[{songName}] Calling response_obj.raise_for_status()...")
        response_obj.raise_for_status() 
        logger.debug(f"[{songName}] response_obj.raise_for_status() successful.")

        logger.debug(f"[{songName}] Accessing response_obj.content...")
        audio_data_content = response_obj.content # 首先直接访问 .content
        logger.debug(f"[{songName}] response_obj.content accessed, length: {len(audio_data_content) if audio_data_content is not None else 'None'}. Now creating BytesIO.")
        
        logger.debug(f"[{songName}] Creating BytesIO object...")
        audio_data = BytesIO(audio_data_content)
        logger.debug(f"[{songName}] BytesIO object created.")

        # 这条日志之前出现得很晚
        logger.info(f"歌曲 {songName} 下载成功，大小: {len(audio_data.getvalue())} bytes") 

        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_file_name_for_this_attempt = temp_file.name
        current_temp_file = temp_file_name_for_this_attempt
        
        temp_file.write(audio_data.getbuffer())
        temp_file.close() 
        
        logger.debug(f"临时音频文件已创建: {temp_file_name_for_this_attempt}")

        pygame.mixer.music.load(temp_file_name_for_this_attempt)
        logger.debug(f"'{temp_file_name_for_this_attempt}' 已加载到pygame mixer")
        pygame.mixer.music.play()
        logger.debug("调用 pygame.mixer.music.play()")
        
        time.sleep(0.5) 
        if pygame.mixer.music.get_busy():
            playing = True
            pause = False
            logger.info(f"成功开始播放: {songName}")
            return songName
        else:
            logger.error(f"Pygame未能开始播放歌曲: {songName} (mixer.music.get_busy() is False)")
            playing = False; pause = False
            if temp_file_name_for_this_attempt and os.path.exists(temp_file_name_for_this_attempt):
                try:
                    os.remove(temp_file_name_for_this_attempt)
                    logger.debug(f"清理播放失败的临时文件: {temp_file_name_for_this_attempt}")
                    if current_temp_file == temp_file_name_for_this_attempt:
                         current_temp_file = None
                except Exception as del_e:
                    logger.error(f"清理播放失败的临时文件时出错: {del_e}")
            return downloadAndPlay(music_json, index + 1)
        
    except requests.exceptions.RequestException as req_e:
        logger.error(f"下载歌曲 {songName} (ID: {songid}) 失败: {req_e}")
        playing=False; pause = False
        if temp_file_name_for_this_attempt and os.path.exists(temp_file_name_for_this_attempt):
             try: os.remove(temp_file_name_for_this_attempt); logger.debug(f"清理下载失败的临时文件: {temp_file_name_for_this_attempt}"); current_temp_file = None if current_temp_file == temp_file_name_for_this_attempt else current_temp_file
             except Exception as del_e: logger.error(f"清理下载失败临时文件时出错: {del_e}")
        return downloadAndPlay(music_json,index+1)
        
    except pygame.error as pg_e:
        logger.error(f"Pygame播放 {songName} (来自 {temp_file_name_for_this_attempt}) 时出错: {pg_e}")
        playing=False; pause = False
        if temp_file_name_for_this_attempt and os.path.exists(temp_file_name_for_this_attempt):
             try: os.remove(temp_file_name_for_this_attempt); logger.debug(f"清理Pygame错误的临时文件: {temp_file_name_for_this_attempt}"); current_temp_file = None if current_temp_file == temp_file_name_for_this_attempt else current_temp_file
             except Exception as del_e: logger.error(f"清理Pygame错误临时文件时出错: {del_e}")
        return downloadAndPlay(music_json,index+1)
        
    except Exception as e:  
        logger.error(f"处理歌曲 {songName} (ID: {songid}) 时发生通用错误: {e}", exc_info=True)
        playing=False; pause = False
        if temp_file_name_for_this_attempt and os.path.exists(temp_file_name_for_this_attempt):
             try: os.remove(temp_file_name_for_this_attempt); logger.debug(f"清理通用错误的临时文件: {temp_file_name_for_this_attempt}"); current_temp_file = None if current_temp_file == temp_file_name_for_this_attempt else current_temp_file
             except Exception as del_e: logger.error(f"清理通用错误临时文件时出错: {del_e}")
        return downloadAndPlay(music_json,index+1)

@mcp.tool()       
def getPlaybackStatus():
    """
    检查当前的播放状态。
    
    返回: 
        "playing" 如果正在播放,
        "paused" 如果已暂停,
        "stopped" 如果已停止或未播放.
    """
    global playing, pause
    logger.debug("getPlaybackStatus 调用")
    current_status = "stopped" # 默认状态
    try:
        pygame_busy = pygame.mixer.music.get_busy()
        logger.debug(f"pygame_busy={pygame_busy}, global playing={playing}, global pause={pause}")

        if playing: # 全局标志认为正在播放或暂停
            if pause: # 全局标志认为是暂停
                # 如果全局标志是 playing=True, pause=True
                # 即使pygame不忙，也应视为"paused"，因为它是用户显式暂停的
                current_status = "paused"
                if not pygame_busy:
                    logger.warning("状态观察：全局为 playing=True, pause=True, 但pygame不忙。仍视为 'paused'。")
            else: # 全局标志认为是正在播放 (playing=True, pause=False)
                if pygame_busy:
                    current_status = "playing"
                else:
                    # 全局标志是播放，但pygame不忙 -> 状态不一致
                    logger.warning("状态不同步：全局标志为播放，但pygame不忙。重置状态为 'stopped'。")
                    playing = False # 纠正全局标志
                    # pause 已经是 False
                    current_status = "stopped"
        else: # 全局标志认为已停止 (playing=False)
            if pygame_busy:
                # 全局标志是停止，但pygame仍然在忙 -> 状态不一致
                logger.warning("状态严重不同步：全局标志为停止，但pygame仍在忙。强制停止并重置。")
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    logger.debug("getPlaybackStatus: 已强制停止并卸载音乐。")
                except Exception as e:
                    logger.error(f"getPlaybackStatus: 强制停止pygame时出错: {e}")
                # playing 已经是 False
                pause = False # 确保 pause 也为 False
                current_status = "stopped"
            else:
                # 全局标志是停止 (playing=False)，pygame也不忙 -> 状态一致为停止
                # 确保 pause 也为 False，因为停止状态不应该保留暂停状态
                if pause:
                    logger.debug("getPlaybackStatus: playing is False, 但 pause is True。重置 pause 为 False。")
                    pause = False
                current_status = "stopped"
        
        logger.info(f"返回播放状态: {current_status}")
        return json.dumps({"status": current_status})
            
    except Exception as e:
        logger.error(f"检查播放状态时发生严重错误: {e}", exc_info=True)
        # 发生未知错误时，重置状态为停止，以确保安全
        playing = False 
        pause = False
        return json.dumps({"status": "stopped", "error": "Error getting status: " + str(e)}) # 确保返回JSON字符串
           

@mcp.tool()      
def stopplay():
    """
    停止播放音乐
    
    返回: 
        播放状态: 已停止
    """
    global playing, pause, current_temp_file
    logger.info("stopplay: 收到停止指令")
    try:
        logger.debug("执行 pygame.mixer.music.stop()")
        pygame.mixer.music.stop()  
        pygame.mixer.music.unload() # 卸载音乐以释放文件
        logger.debug("音乐已停止并卸载")
        time.sleep(0.1) # 给卸载操作一点时间
        
        # 再次检查是否繁忙，以防万一
        if pygame.mixer.music.get_busy(): 
            logger.warning("调用stop和unload后，pygame仍然报告繁忙，尝试 fadeout...")
            pygame.mixer.music.fadeout(500) 
            pygame.mixer.music.unload() # fadeout 后再次确保卸载
            time.sleep(0.5) # 给fadeout和卸载时间
            if pygame.mixer.music.get_busy():
                 logger.error("再次停止和卸载后pygame仍然繁忙！音乐可能未完全停止。")
        else:
            logger.debug("stopplay: Pygame mixer确认已停止且音乐已卸载。")

        # 只有在确认mixer不繁忙后才尝试删除文件
        if not pygame.mixer.music.get_busy() and current_temp_file and os.path.exists(current_temp_file):
            try:
                logger.debug(f"准备清理临时文件: {current_temp_file}")
                os.remove(current_temp_file)
                logger.info(f"stopplay: 清理了临时文件 {current_temp_file}")
            except Exception as e:
                logger.error(f"stopplay: 清理临时文件 {current_temp_file} 时出错: {e}")
            finally:
                # 无论删除是否成功，都将 current_temp_file 设为 None，因为它要么被删了，要么删除失败不应再用
                current_temp_file = None 
        elif current_temp_file:
            logger.warning(f"stopplay: Mixer仍在忙或文件不存在 ({current_temp_file})，未尝试删除临时文件。")
        else:
            logger.debug("stopplay: 没有当前临时文件需要清理或文件不存在。")

    except Exception as e:
        logger.error(f"stopplay: 执行pygame.mixer.music.stop()或相关操作时出错: {e}", exc_info=True)
    
    finally: 
        playing = False
        pause = False
        logger.info("stopplay: 全局状态已更新为 playing=False, pause=False")
    
    return json.dumps({"status": "stopped"}) # 确保返回JSON字符串

@mcp.tool()   
def pauseplay():
    """
    暂停音乐播放
    
    返回: 
        播放状态: "paused"
    """
    global playing, pause
    logger.info("pauseplay 调用")
    
    current_playback_status = "unknown"
    try:
        # 先获取当前真实状态，避免错误操作
        raw_status_json = getPlaybackStatus() # 调用我们自己的状态获取函数
        status_dict = json.loads(raw_status_json)
        current_playback_status = status_dict.get("status", "stopped")
        logger.debug(f"pauseplay: 当前通过getPlaybackStatus获取的状态是 {current_playback_status}")
    except Exception as e:
        logger.error(f"pauseplay: 获取播放状态时出错: {e}")
        # 如果获取状态失败，谨慎起见，不执行任何操作，并返回错误
        return json.dumps({"status": "error", "message": "无法获取当前播放状态"})

    if current_playback_status == "playing":
        try:
            pygame.mixer.music.pause()
            pause = True # playing 保持 True
            logger.info("音乐已暂停")
            return json.dumps({"status": "paused"})
        except Exception as e:
            logger.error(f"pygame.mixer.music.pause() 执行失败: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    elif current_playback_status == "paused":
        logger.info("音乐已经是暂停状态")
        return json.dumps({"status": "paused"})
    else: # "stopped" or "unknown" or "error"
        logger.warning(f"音乐未在播放 (状态: {current_playback_status})，无法暂停")
        return json.dumps({"status": "stopped"}) # 返回stopped更合适

@mcp.tool()   
def unpauseplay():
    """
    恢复音乐播放
    
    返回：
        播放状态: "playing"
    """
    global playing, pause
    logger.info("unpauseplay 调用")
    
    current_playback_status = "unknown"
    try:
        raw_status_json = getPlaybackStatus()
        status_dict = json.loads(raw_status_json)
        current_playback_status = status_dict.get("status", "stopped")
        logger.debug(f"unpauseplay: 当前通过getPlaybackStatus获取的状态是 {current_playback_status}")
    except Exception as e:
        logger.error(f"unpauseplay: 获取播放状态时出错: {e}")
        return json.dumps({"status": "error", "message": "无法获取当前播放状态"})

    if current_playback_status == "paused":
        try:
            pygame.mixer.music.unpause()
            pause = False # playing 保持 True
            logger.info("音乐已恢复播放")
            return json.dumps({"status": "playing"})
        except Exception as e:
            logger.error(f"pygame.mixer.music.unpause() 执行失败: {e}")
            return json.dumps({"status": "error", "message": str(e)})
    elif current_playback_status == "playing":
        logger.info("音乐已在播放状态")
        return json.dumps({"status": "playing"})
    else: # "stopped" or "unknown" or "error"
        logger.warning(f"音乐未暂停或已停止 (状态: {current_playback_status})，无法恢复")
        return json.dumps({"status": "stopped"})

    
if __name__ == "__main__":
   logger.info(f"音乐播放器服务 (mcp_server_onlinemusic_player.py) 正在启动，PID: {os.getpid()}")
   mcp.run(transport='stdio')
   logger.info("音乐播放器服务已停止") # 这句可能在正常stdio退出时不会执行