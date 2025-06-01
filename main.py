# -*- coding: utf-8 -*-
import asyncio
import base64
import gzip
import json
import logging
import os
import platform
import random
import re
import time
import traceback
import warnings
from datetime import datetime
from typing import Optional
# import sqlite3

import bilibili_api
import httpx
import jieba
# from gradio_client import Client as GradioClient
import psutil
import schedule
import websockets
import yaml
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from selenium.webdriver import (Chrome, ChromeOptions, Edge, EdgeOptions,
                                Firefox, FirefoxOptions)
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
from wordcloud import WordCloud
import fnmatch

APP_VERSION = "0.2.0.2"
SECURITY_LOGGER_LEVEL = 35

def init_structure():
    """初始化程序文件目录结构

    :return: None
    """
    os.makedirs("data/chat/group", exist_ok=True)
    os.makedirs("data/chat/user", exist_ok=True)
    os.makedirs("data/wordcloud", exist_ok=True)
    os.makedirs("data/wordcloud", exist_ok=True)
    os.makedirs('data/web_screenshot/drivers', exist_ok=True)
    os.makedirs('cache/lolicon_api', exist_ok=True)
    os.makedirs('cache/wordcloud', exist_ok=True)
    os.makedirs('cache/web_screenshot', exist_ok=True)
    os.makedirs('fonts', exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def calc_uptime(start: int | float, current: int | float = time.time(), enforce_int_output: bool = False) -> tuple:
    """计算上线时间

    :param start: 启动时时间
    :param current: 当前时间
    :param enforce_int_output: 是否输出为整数
    :return: 天, 小时, 分钟, 秒, 毫秒
    """
    elapsed_time = current - start
    days, remainder = divmod(elapsed_time, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = seconds % 1 * 1000

    if enforce_int_output:
        return int(days), int(hours), int(minutes), int(seconds), int(milliseconds)
    else:
        return days, hours, minutes, seconds, milliseconds


def compress_latest_logs(path: os.PathLike | str = 'logs/main.log') -> None:
    """压缩旧的运行日志（如果存在）

    path -> path + yyyy-MM-dd.gz
    
    :param path: 日志文件路径
    :return: None
    """
    # 当路径为块设备时（如`/dev/null`），则不进行压缩
    if os.path.ismount(path):
        return

    elif os.path.exists(path):
        print("Compressing old logs...")

        timestamp = datetime.now().strftime("%Y-%m-%d")

        # 将日志保存至原目录，然后重命名
        compressed_path = os.path.join(os.path.dirname(path), f"{timestamp}.log.gz")

        # 如果已有相同名称的文件，则保存为.n
        # 然后再次检查，直到不重复为止
        counter = 0
        original_compressed_path = compressed_path
        while os.path.exists(compressed_path):
            counter += 1
            compressed_path = f"{original_compressed_path[:-3]}.{counter}.gz"

        try:
            with open(path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    # 使用二进制模式读取和写入
                    f_out.write(f_in.read())
            os.remove(path)
        except Exception as e:
            warnings.warn(f"Error compressing logs: {e}", ResourceWarning)
            # 如果压缩失败，保留原始文件
            return
    else:
        return

def get_cpu_name() -> str:
    """获取CPU名称，如果无法获取则返回Unknown CPU

    :return: str
    """
    if platform.system() == 'Windows':
        return platform.processor()
    elif platform.system() == 'Linux':
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    return line.split(':')[1].strip()
    return 'Unknown CPU'


# 异常类
class WrongVerifyKeyException(Exception):
    pass


class BotNotExistException(Exception):
    pass


class SessionInvalidException(Exception):
    pass


class TargetNotExistException(Exception):
    pass


class MessageCancelledException(Exception):
    pass

# 创建Logger类
class CustomLogger(logging.Logger):
    def security(self, msg, *args, **kwargs):
        """
        添加security方法，用于记录安全相关的日志
        """
        if self.isEnabledFor(SECURITY_LOGGER_LEVEL):
            self._log(SECURITY_LOGGER_LEVEL, msg, args, **kwargs)

class CallableDict:
    def __init__(self, data: dict):
        """初始化可调用字典对象

        :param data: 要包装的字典数据
        """
        self._data = data

    def __getattr__(self, item):
        """通过属性访问字典值

        :param item: 要访问的键名
        :return: 如果值是字典则返回新的CallableDict实例，否则返回原始值
        """
        if type(self._data[item]) is dict:
            return CallableDict(self._data[item])
        else:
            return self._data[item]

    def __getitem__(self, key):
        """通过索引访问字典值

        :param key: 要访问的键名
        :return: 对应的值
        """
        return self._data[key]

    def get_original(self) -> dict:
        """获取原始字典数据

        :return: 原始字典数据
        """
        return self._data


class Config:
    def __init__(self, path: os.PathLike | str = 'config.yml'):
        """初始化配置对象

        :param path: 配置文件路径，默认为'config.yml'
        :raises RuntimeError: 当配置文件不存在时
        :raises yaml.YAMLError: 当配置文件格式错误时
        """
        try:
            with open(path, "r", encoding="utf-8") as cf:
                self._data = yaml.load(cf, yaml.FullLoader)
        except FileNotFoundError:
            raise RuntimeError("Config file not found.")
        except yaml.YAMLError:
            raise yaml.YAMLError("Config file format ERROR")

    def __getitem__(self, key):
        """通过索引访问配置值

        :param key: 要访问的配置键名
        :return: 对应的配置值
        """
        return self._data[key]

    def __getattr__(self, item) -> CallableDict:
        """通过属性访问配置值

        :param item: 要访问的配置键名
        :return: 如果值是字典则返回CallableDict实例，否则返回原始值
        """
        if type(self._data[item]) is dict:
            return CallableDict(self._data[item])
        else:
            return self._data[item]

    def get_data(self):
        """获取原始配置数据

        :return: 原始配置字典
        """
        return self._data


class Lang:
    def __init__(self, path: os.PathLike | str = None):
        """初始化语言对象

        :param path: 语言文件路径，如果为None则使用默认的'lang.yml'
        :raises RuntimeError: 当语言文件不存在时
        :raises yaml.YAMLError: 当语言文件格式错误时
        """
        if not path:
            logger.warning('No language file specified')
            path = 'lang.yml'

        try:
            with open(path, "r", encoding="utf-8") as cf:
                self._data = yaml.load(cf, yaml.FullLoader)
        except FileNotFoundError:
            raise RuntimeError("Language file not found.")
        except yaml.YAMLError:
            raise yaml.YAMLError("Language file format ERROR")

    def __getitem__(self, key):
        """通过索引访问语言文本

        :param key: 要访问的语言键名
        :return: 对应的语言文本，如果键不存在则返回错误提示
        """
        if key not in self._data:
            return f"Language key missing: {key}"
        return self._data[key]

    def __getattr__(self, item):
        """通过属性访问语言文本

        :param item: 要访问的语言键名
        :return: 对应的语言文本，如果键不存在则返回错误提示
        """
        if item not in self._data:
            return f"Language key missing: {item}"
        elif type(self._data[item]) is dict:
            return CallableDict(self._data[item])
        else:
            return self._data[item]

    def get_data(self):
        """获取原始语言数据

        :return: 原始语言字典
        """
        return self._data


class WebScreenshotService:
    def __init__(self, browser: str):
        """初始化浏览器实例

        :param browser: 浏览器类型，必须为`Chrome`、`Firefox`或`Edge`（大小写不敏感）
        """
        self.browser_name = browser

        if self.browser_name.lower() == 'chrome':
            # 设置chrome为无头模式
            self.options = ChromeOptions()
            self.options.add_argument('--headless')
            self.options.add_argument('--start-maximized')
            self.service = ChromeService(ChromeDriverManager().install())
        elif self.browser_name.lower() == 'firefox':
            # 设置firefox为无头模式
            self.options = FirefoxOptions()
            self.options.add_argument('--headless')
            self.service = FirefoxService(GeckoDriverManager().install())
        elif self.browser_name.lower() == 'edge':
            # 设置edge为无头模式
            self.options = EdgeOptions()
            self.options.add_argument('--headless')
            self.options.add_argument('--start-maximized')
            self.service = EdgeService(EdgeChromiumDriverManager().install())
        else:
            raise ValueError('Invalid browser type')

    def __delete__(self, instance):
        """

        :param instance:
        :return:
        """
        self.close()

    def get_driver(self):
        """获取WebDriver实例

        :return: WebDriver实例
        """
        if self.browser_name.lower() == 'chrome':
            return Chrome(service=self.service, options=self.options)
        elif self.browser_name.lower() == 'firefox':
            return Firefox(service=self.service, options=self.options)
        elif self.browser_name.lower() == 'edge':
            return Edge(service=self.service, options=self.options)

    def close(self):
        """关闭Service

        :return: None
        """
        self.service.stop()


class WebScreenshotBrowser:
    def __init__(self, driver):
        """初始化截图浏览器

        :param driver: WebDriver实例
        """
        self.driver = driver

    def __delete__(self, instance):
        """删除浏览器实例

        :param instance:
        :return:
        """
        self.close()

    async def capture_full_page_screenshot(self, url, output_file, implicitly_wait_time: float | int = 10) -> bool:
        """捕获整个网页的截图

        :param url: 网页URL
        :param output_file: 输出文件路径
        :param implicitly_wait_time: 隐式等待时间
        :return: bool 截图是否成功
        """
        logger.info(app_lang.logs.web_screenshot.capturing.format(url=url))

        # 访问指定的URL
        self.driver.get(url)

        # 等待一段时间，确保渲染完成
        self.driver.implicitly_wait(implicitly_wait_time)

        # 获取网页的长和宽
        required_width = self.driver.execute_script('return document.body.parentNode.scrollWidth')
        required_height = self.driver.execute_script('return document.body.parentNode.scrollHeight')

        # 设置浏览器窗口的大小
        self.driver.set_window_size(required_width, required_height)

        # 获取DOM元素
        element = self.driver.find_element(By.TAG_NAME, "body")

        # 截图并保存
        return element.screenshot(output_file)

    def close(self):
        # 关闭浏览器
        self.driver.quit()


def on_message_types(*message_types):
    """当消息类型匹配列表中任一类型时执行

    :param message_types: 可接受的消息类型列表
    :return:
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 假设 MiraiResponse 对象是 args 的第二个参数
            mirai_res = args[1]
            message_type = mirai_res.get_message_type()

            if message_type in message_types:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def include_bot_at():
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 假设 MiraiResponse 对象是 args 的第二个参数
            mirai_res = args[1]
            message_type = mirai_res.get_message_type()

            # 如果消息类型是 FriendMessage，直接执行函数
            if message_type == 'FriendMessage':
                return await func(*args, **kwargs)

            # 对于 GroupMessage，检查是否有提及机器人的部分
            if message_type == 'GroupMessage':
                message_chain = mirai_res.get_message_chain()
                for message in message_chain:
                    if message['type'] == 'At' and message['target'] == bot.account:
                        # 如果找到提及机器人的消息，执行原函数
                        return await func(*args, **kwargs)

            # 如果没有提及机器人，或者消息类型不是 GroupMessage 或 FriendMessage，不执行原函数
            return None

        return wrapper

    return decorator


def no_command_prefix():
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 假设 MiraiResponse 对象是 args 的第二个参数
            mirai_res = args[1]
            message = mirai_res.__str__()

            # 如果消息链中没有命令前缀，执行原函数
            if not message.startswith(app_config.app.commands.prefix):
                return await func(*args, **kwargs)

            # 如果有命令前缀，则不执行原函数
            return None

        return wrapper

    return decorator


class MyScheduler(object):
    class BuiltinScheduledTasks:
        @staticmethod
        async def send_hitokoto(message_type: str, targets: list[int], prefix: str = '', suffix: str = '',
                                send_delay: int | float = 1, url_options: str = '') -> None:
            # mirai_req = MiraiRequest()
            try:
                r = httpx.get('https://v1.hitokoto.cn/' + url_options)
                r.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(str(e))

                if app_config.bot.send_exception:
                    # 批量发送错误消息
                    for target in targets:
                        await bot.send('send' + message_type, target, [
                            {
                                "type": "Plain",
                                "text": str(e)
                            }
                        ])
                        await asyncio.sleep(send_delay)
                return
            else:
                for target in targets:
                    await bot.send('send' + message_type, target, [
                        {
                            "type": "Plain",
                            "text": prefix + '%s\n——%s' % (r.json()['hitokoto'], r.json()['from']) + suffix
                        }
                    ])
                    await asyncio.sleep(send_delay)

    def __init__(self):
        """计划任务类

        """

        for task in app_config.schedule.tasks:
            if task['type'] == 'interval_seconds' or task['type'] == 'interval':
                schedule.every(task['time']).seconds.do(exec, task['exec'], globals(), locals())
            elif task['type'] == 'interval_minutes':
                schedule.every(task['time']).minutes.do(exec, task['exec'], globals(), locals())
            elif task['type'] == 'daily':
                schedule.every().day.at(task['time']).do(exec, task['exec'], globals(), locals())
            else:
                logger.warning(
                    app_lang.logs.schedule.unknown_task.format(type=task['type']))
            logger.debug(app_lang.logs.schedule.task_created.format(
                type=task['type'], exec=task['exec'])
            )

    async def async_run_pending(self):
        """

        :return: None
        """
        logger.debug(app_lang.logs.schedule.schedule_checked)
        return schedule.run_pending()


class OpenAIChat(object):
    def __init__(self, qq: int, is_group: bool = False):
        """
        初始化会话类
        :param qq: 用户/群的QQ号
        """
        self.qq = qq
        self.db = []
        self.is_group = is_group

        # 检查是否为群聊
        if self.is_group:
            self.db_path = 'data/chat/group/%s.json' % qq
        else:
            self.db_path = 'data/chat/user/%s.json' % qq

        self.lock = asyncio.Lock()
        self.client = OpenAI(
            api_key=app_config.openai_chat.api_key,
            base_url=app_config.openai_chat.base_url,
        )

    async def load_history(self):
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
        except FileNotFoundError:
            logger.info(app_lang.logs.openai_chat.history_not_found)

    async def save_history(self):
        """
        保存聊天历史记录
        """
        try:
            async with self.lock:
                with open(self.db_path, "w", encoding="utf-8") as f:
                    json.dump(self.db, f, ensure_ascii=False)
            logger.debug(app_lang.logs.openai_chat.history_saved)
        except Exception as e:
            logger.error(app_lang.logs.openai_chat.history_save_error.format(e=e))

    async def check_target_db_exist(self, qq: Optional[int] = None, is_group: Optional[bool] = None) -> bool:
        """

        :param is_group:
        :param qq: 用户的QQ号
        :return: bool
        """
        async with self.lock:  # 使用异步锁
            if qq is None:  # 如果没有指定QQ号，则使用初始化时的QQ号
                qq = self.qq

            if is_group is None:  # 如果没有指定是否为群聊，则使用初始化时的值
                is_group = self.is_group

            if is_group:
                db_path = 'data/chat/group/%s.json' % qq
            else:
                db_path = 'data/chat/user/%s.json' % qq

            return os.path.exists(db_path)

    async def create(self, present: Optional[list] = None):
        """创建一个新的会话

        :param present: 预设对话内容
        :return: None
        """
        async with self.lock:  # 使用异步锁
            # 替换可变实参值
            if present is None:
                # 使用默认AI
                self.db = app_config.openai_chat.presents.default.conversations
            else:
                # 使用指定预设
                self.db = present

    async def generate(self) -> str:
        """生成响应

        :return: 响应内容
        """
        async with self.lock:  # 使用异步锁
            completion = self.client.chat.completions.create(
                model=app_config.openai_chat.model, messages=self.db
            )
        content = completion.choices[0].message.content
        await self.append(content=content, who="assistant")
        return content

    async def append(self, content: str, who: str):
        """添加对话内容

        :param content: 对话内容
        :param who: 对话来源，可以是'user'或'assistant'
        :return: None
        """
        if who not in ["user", "assistant"]:
            raise ValueError()

        async with self.lock:  # 使用异步锁
            self.db.append({"role": who, "content": content})


class MinigameGobang(object):
    def __init__(self):
        self.players = []
        self.board_cols = 9
        self.board_rows = 9
        self.board = [[0 for _ in range(self.board_cols)] for _ in range(self.board_rows)]

    def add_player(self, qq: int, player_sign: str):
        pass

    def check_winner(self):
        n = len(self.board)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平，垂直，对角线向下，对角线向上

        def in_bounds(_x, _y):
            return 0 <= _x < n and 0 <= _y < n

        def check_line(_x, _y, _dx, _dy, _player):
            count = 0
            while in_bounds(_x, _y) and self.board[_x][_y] == _player:
                count += 1
                _x += _dx
                _y += _dy
            return count >= 5

        for x in range(n):
            for y in range(n):
                if self.board[x][y] != 0:
                    player = self.board[x][y]
                    for dx, dy in directions:
                        if check_line(x, y, dx, dy, player):
                            return player
        return 0

    # 示例棋盘
    # board = [
    #     [0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0]
    # ]
    #
    # print(check_winner(board))  # 输出 1


class MiraiMessageBase(object):
    def __str__(self, use_strip: bool = False, stringifies_at: bool = True) -> str:
        """转换消息链为纯文本。无法用纯文本表示的消息会丢失信息，如任何图片都是`[图片]`
        https://github.com/mamoe/mirai/blob/dev/docs/Messages.md#%E6%B6%88%E6%81%AF%E5%85%83%E7%B4%A0

        :param stringifies_at: 是否转换At信息
        :param use_strip: 是否使用strip
        :return: str
        """
        _text = ''

        for msg in self.get_message_chain():
            # 如果为纯文本
            if msg['type'] == 'Plain':
                _text += msg["text"]
            elif msg['type'] == 'Image':
                _text += '[图片]'
            elif msg['type'] == 'At':
                if stringifies_at:
                    _text += '@%s' % msg['target']
            elif msg['type'] == 'AtAll':
                _text += '@全体成员'
            elif msg['type'] == 'File':
                _text += '[文件]'

        if use_strip:
            _text = _text.strip()

        return _text


class MiraiResponse(MiraiMessageBase):
    def __init__(self, full_data_structure: dict):
        """初始化Mirai消息基类

        :param full_data_structure: Mirai返回的完整消息结构
        """
        super().__init__()

        self.sync_id = full_data_structure.get('syncId', '-1')
        self.data = full_data_structure.get('data', {})
        self.content = full_data_structure.get('content', {})

    def get_sync_id(self) -> str:
        """获取syncId

        :return: str
        """
        return self.sync_id

    def get_message_chain(self) -> list:
        """获取消息链

        :return: list
        """
        # 在data或content中查找messageChain
        if 'messageChain' in self.data:
            return self.data.get('messageChain', [])
        elif 'messageChain' in self.content:
            return self.content.get('messageChain', [])
        else:
            return []

    def format_msg(self) -> str:
        """格式化消息

        :return: str
        """
        message_type = self.get_message_type()

        # 如果有code字段，说明是响应
        if self.get_response_code() is not None:
            return app_lang.logs.format_msg.response.format(
                code=self.get_response_code()
            )

        # 根据消息类型进行判断
        elif message_type == 'Auth':
            return app_lang.logs.format_msg.auth.format(
                session=self.get_session()
            )
        elif message_type == 'GroupMessage':
            return app_lang.logs.format_msg.group_message.format(
                group_name=self.get_group_name(), group_id=self.get_group_id(),
                user_name=self.get_sender_name()[0], user_id=self.get_sender_id(),
                content=self.__str__()
            )
        elif message_type == 'FriendMessage':
            return app_lang.logs.format_msg.friend_message.format(
                user_name=self.get_sender_name()[0], user_id=self.get_sender_id(),
                content=self.__str__()
            )
        elif message_type == 'GroupRecallEvent':
            return app_lang.logs.format_msg.recall_message.format(
                user_name=self.get_sender_name()[0], user_id=self.get_sender_id(),
                message_id=self.data['messageId']
            )
        elif message_type == 'NudgeEvent':
            return app_lang.logs.format_msg.nudge_message.format(
                user_id=self.get_sender_id(),
                action=self.get_nudge_action(), target_id=self.get_nudge_target_id(),
                suffix=self.get_nudge_suffix()
            )
        else:
            return app_lang.logs.format_msg.unknown_message.format(
                content=self.__str__()
            )

    def get_response_code(self) -> int | None:
        """获取响应代码

        :return: int | None
        """
        return self.data.get('code', None)

    def get_session(self) -> str | None:
        """获取session

        注：理论上只能在注册客户端时使用，之后session会保存到Bot.session中

        :return: str | None
        """
        return self.data.get('session', None)

    def get_message_type(self) -> str | None:
        """获取消息类型

        :return: str
        """
        if 'code' in self.data:
            return 'Response'
        elif 'session' in self.data:
            return 'Auth'
        else:
            return self.data.get('type', None)

    def get_sender_id(self) -> int:
        """获取发送者的QQ号

        :return: int
        """
        message_type = self.get_message_type()

        # 然后进行判断
        if message_type in ['GroupMessage', 'FriendMessage']:  # 最常见的类型
            return int(self.data['sender']['id'])
        elif message_type == 'GroupRecallEvent':  # 群聊撤回事件
            return int(self.data['operator']['id'])
        elif message_type == 'NudgeEvent':  # 戳一戳事件
            return int(self.data['fromId'])
        else:
            return 0

    def get_group_id(self) -> int | None:
        """获取群号

        :return: int | None
        """
        message_type = self.get_message_type()

        if message_type == 'GroupMessage':
            return self.data['sender']['group']['id']
        else:
            return None

    def get_group_name(self) -> str | None:
        """获取群名称

        :return: str | None
        """
        message_type = self.get_message_type()

        if message_type == 'GroupMessage':
            return self.data['sender']['group']['name']
        else:
            return None

    def get_sender_name(self) -> tuple[str | None, str | None]:
        """获取事件发送者的名称

        :return: tuple[用户实际名字, 用户备注/别名]
        """
        message_type = self.get_message_type()

        if message_type == 'GroupMessage':
            return self.data['sender']['memberName'], self.data['sender']['specialTitle']
        elif message_type == 'FriendMessage':
            return self.data['sender']['nickname'], self.data['sender']['remark']
        elif message_type == 'GroupRecallEvent':
            return self.data['operator']['memberName'], self.data['operator']['specialTitle']
        else:
            return None, None

    def get_nudge_target_id(self) -> int | None:
        """获取戳一戳目标的QQ号

        :return: int | None
        """
        message_type = self.get_message_type()

        if message_type == 'NudgeEvent':
            return self.data['target']
        else:
            return None

    def get_nudge_action(self) -> str | None:
        """获取戳一戳动作

        :return: str | None
        """
        message_type = self.get_message_type()

        if message_type == 'NudgeEvent':
            return self.data['action']
        else:
            return None

    def get_nudge_suffix(self) -> str | None:
        """获取戳一戳后缀

        :return: str | None
        """
        message_type = self.get_message_type()

        if message_type == 'NudgeEvent':
            return self.data['suffix']
        else:
            return None

    def get_target_loc(self) -> int | None:
        """快速获取机器人发送消息的目标

        :return: int | None
        """
        return self.get_group_id() if self.get_group_id() else self.get_sender_id()

    def is_in_user_list(self, user_list: list[int]) -> bool:
        """检查发送者是否在用户列表中

        :param user_list: 用户列表
        :return: bool
        """
        return self.get_sender_id() in user_list

    def is_in_group_list(self, group_list: list[int]) -> bool:
        """检查发送者是否在群组列表中

        :param group_list: 群组列表
        :return: bool
        """
        return self.get_group_id() in group_list if self.get_group_id() else False


class MiraiRequest(MiraiMessageBase):
    def __init__(self, command: str, sub_command: Optional[str] = None,
                 message_chain=None):
        """ 初始化Mirai请求基类

        :param command:
        :param sub_command:
        :param message_chain:
        """
        super().__init__()

        self.sync_id = '-1'
        self.session = None
        self.command = command
        self.sub_command = sub_command

        # 消息链
        if message_chain is None:
            self.message_chain = []

        # 是否取消消息
        self.is_cancelled = False

        self.target = None

    def from_resp(self, mirai_res: MiraiResponse):
        """

        :param mirai_res: MiraiResponse
        :return: None
        """
        self.target = mirai_res.get_target_loc()

    def set_session(self, session: str):
        """

        :param session:
        :return: None
        """
        self.session = session

    def set_target_loc(self, target: int):
        """

        :param target:
        :return: None
        """
        self.target = target

    def get_target_loc(self) -> int | None:
        """

        :return: int | None
        """
        return self.target

    def add_plain(self, text: str):
        """添加纯文本消息

        :param text: 纯文本消息
        :return: None
        """
        self.message_chain.append({
            "type": "Plain",
            "text": text,
        })

    def add_image(self, image_id: str = '', url: str = '', path: str = None, _base64: str = None, width: int = 0,
                  height: int = 0, size: int = 0, image_type: str = 'UNKNOWN', is_emoji: bool = False):
        """添加图片消息
        * 注意：在`image_id`，`url`，`path`，`_base64`中需要至少提供一个参数
          但推荐只使用其中的一个参数

        :param image_id: 图片ID
        :param url: 图片URL
        :param path: 图片文件路径（相对于MiraiConsole的工作目录）
        :param _base64: 图片的Base64编码
        :param width: 图片宽度
        :param height: 图片高度
        :param size: 图片大小
        :param image_type: 图片类型
        :param is_emoji: 是否为表情
        :return: None
        """
        # 检测是否提供了至少一个参数
        if image_id is None and url is None and path is None and _base64 is None:
            raise ValueError
        else:
            # 添加图片消息
            self.message_chain.append({
                "type": "Image",
                "imageId": image_id,
                "url": url,
                "path": path,
                "base64": _base64,
                "width": width,
                "height": height,
                "size": size,
                "imageType": image_type,
                "isEmoji": is_emoji
            })

    def add_custom_content(self, content: dict):
        """添加自定义内容

        :param content: 自定义内容
        :return: None
        """
        self.message_chain.append(content)

    def reset_message_chain(self):
        """重置消息链

        :return: None
        """
        self.message_chain = []

    def get_message_chain(self) -> list:
        """获取消息链

        :return: list
        """
        return self.message_chain

    def cancel(self, raise_exception: bool = True) -> None:
        """取消消息

        :return: None
        """
        if raise_exception:
            raise MessageCancelledException
        else:
            self.is_cancelled = True
            return None

    def dump_payload(self, is_json: bool = True) -> str | dict:
        """

        :param is_json: 是否转换为JSON格式
        :return: str | dict
        """
        payload = {
            "syncId": self.sync_id,
            "command": self.command,
            "subCommand": self.sub_command,
            "content": {
                "sessionKey": self.session,
                "target": self.target,
                "messageChain": None if len(self.message_chain) == 0 else self.message_chain,
            },
        }

        if is_json:
            return json.dumps(payload)
        else:
            return payload


class Bot(object):
    def __init__(
            self,
            account: int,
            verify_key: str,
            websocket_url: str = "ws://localhost:8080/all",
    ):
        """Bot类

        :param account: QQ号
        :param verify_key: 验证密钥
        :param websocket_url: WebSocket地址
        """
        self.startup_time = 0
        self.account = account
        self.verify_key = verify_key
        self.websocket_url = websocket_url
        self.session = None
        self.ws = None
        self.scheduler = None

        # 创建Credential对象
        if app_config.bilibili_url_detect.enable:
            self.bilibili_credential = bilibili_api.Credential(
                sessdata=app_config.bilibili_url_detect.credential.sessdata,
                bili_jct=app_config.bilibili_url_detect.credential.bili_jct,
                buvid3=app_config.bilibili_url_detect.credential.buvid3,
                dedeuserid=app_config.bilibili_url_detect.credential.dedeuserid
            )

        # 创建计划任务对象，开始注册计划任务
        if app_config.schedule.enable:
            self.scheduler = MyScheduler()

        # 创建WebScreenshotService对象
        if app_config.auto_web_screenshot.enable:
            self.web_screenshot_service = WebScreenshotService(
                browser=app_config.web_screenshot_common.browser
            )

        # 创建计数器
        self.counter = {
            'recv': 0,
            'send': 0,
            'error': 0
        }

    async def send_mirai_req(self, mirai_req: MiraiRequest):
        if mirai_req.get_message_chain() is None or \
                len(mirai_req.get_message_chain()) == 0 or \
                mirai_req.is_cancelled is True:
            return None  # 如果消息链为空或已取消，则不发送消息

        # 设置session
        mirai_req.set_session(self.session)

        # 记录日志
        logger.info('[SEND-%s] (%s): %s', mirai_req.command, mirai_req.get_target_loc(), mirai_req.__str__())

        self.counter['send'] += 1

        return await self.ws.send(mirai_req.dump_payload(True))

    async def send(
            self,
            command: str,
            target: int,
            message_chain: list,
            sub_command: Optional[str] = None,
    ):
        # 检测消息链是否为None
        if message_chain is None:
            return None

        # 开始构建消息格式
        payload = {
            "syncId": "-1",
            "command": command,
            "subCommand": sub_command,
            "content": {
                "sessionKey": self.session,
                "target": target,
                "messageChain": message_chain,
            },
        }

        reply_json = json.dumps(payload)

        # 记录日志
        logger.info('[SEND-%s] (%s): %s', command, target, MiraiResponse(payload).__str__())

        self.counter['send'] += 1

        return await self.ws.send(reply_json)

    async def recv_raw(self) -> str:
        """

        :return: str
        """
        # 计数器+1
        self.counter['recv'] += 1

        # 返回接收到的消息
        return await self.ws.recv()

    async def recv(self) -> MiraiResponse:
        """

        :return: MiraiResponse
        """
        mirai_res = MiraiResponse(json.loads(await self.recv_raw()))

        # 记录日志
        logger.info('[RECV-%s] %s', str(mirai_res.get_message_type()), mirai_res.format_msg())

        # 返回MiraiResponse对象
        return mirai_res

    @on_message_types('GroupMessage', 'FriendMessage')
    async def _command_base(self, mirai_res: MiraiResponse):
        # 匹配命令前缀
        match = re.match(
            r'^' + app_config.app.commands.prefix + r'(\w+)(?:\s+(.*))?$',
            mirai_res.__str__()
        )
        if match:
            logger.debug(app_lang.logs.found_command_prefix.format(content=mirai_res.__str__()))

            cmd_type = match.group(1)
            args_str = match.group(2) if match.group(2) else ""

            # 支持单引号、双引号和反斜杠转义
            args = []
            current_arg = ""
            in_quote = None
            escape_next = False

            # 解析参数
            for char in args_str:
                if escape_next:  # 如果上一个字符是反斜杠，则当前字符不进行转义
                    current_arg += char
                    escape_next = False
                elif char == '\\':  # 如果当前字符是反斜杠，则下一个字符不进行转义
                    escape_next = True
                elif char in ['"', "'"]:  # 如果当前字符是引号
                    if in_quote is None:  # 如果没有引号，则开始引号
                        in_quote = char
                    elif in_quote == char:  # 如果当前引号与上一个引号相同，则结束引号
                        in_quote = None
                    else:  # 如果当前引号与上一个引号不同，则添加到参数中
                        current_arg += char
                elif char.isspace() and in_quote is None:  # 如果当前字符是空格且不在引号内
                    if current_arg:  # 如果当前参数不为空，则添加到参数列表中
                        args.append(current_arg)
                        current_arg = ""
                else:  # 如果当前字符不是空格，则添加到参数中
                    current_arg += char

            # 添加最后一个参数（包括未闭合的引号内容）
            if current_arg:
                args.append(current_arg)

            # 状态与统计信息
            if (cmd_type == 'stat' or cmd_type == 'status') and app_config.commands.status.enable:
                return await self.send_mirai_req(await self._command_status(mirai_res, True if cmd_type == 'stat' else False))

            # OpenAI聊天（命令控制器）
            elif cmd_type == 'chat' and app_config.openai_chat.enable:
                return await self.send_mirai_req(await self._command_openai_chat(args, mirai_res))

            # lolicon api
            elif cmd_type == 'lolicon' and app_config.commands.lolicon_api.enable:
                return await self.send_mirai_req(await self._command_lolicon_api(args, mirai_res))

            # 随机骰子
            elif cmd_type == 'dice' and app_config.commands.dice.enable:
                return await self.send_mirai_req(await self._command_dice(mirai_res))

            # 词云生成
            elif cmd_type == 'wordcloud' and app_config.wordcloud.enable:
                return await self.send_mirai_req(await self._command_wordcloud(args, mirai_res))

            # 网页截图
            # elif cmd_type == 'wsc':
            #     return await self.send(
            #         'send' + mirai_res.get_message_type(), mirai_res.get_target_loc(),
            #         await self._command_web_screenshot(args, mirai_res)
            #     )

            else:  # 未知命令
                mirai_req = MiraiRequest('send' + mirai_res.get_message_type())
                mirai_req.from_resp(mirai_res)
                mirai_req.add_plain(app_lang.template.unknown_command)
                return await self.send_mirai_req(mirai_req)
        return None

    async def _command_status(self, mirai_res: MiraiResponse, minial: Optional[bool] = True) -> MiraiRequest:
        # 创建MiraiRequest对象
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())
        mirai_req.from_resp(mirai_res)

        if minial:
            mirai_req.add_plain(app_lang.template.status.ping)
        else:
            uptime = calc_uptime(self.startup_time, time.time(), True)
            disk_usage = psutil.disk_usage('.')

            mirai_req.add_plain(app_lang.template.status.detail.format(
                version=APP_VERSION,
                uptime_days=uptime[0], uptime_hours=uptime[1],
                uptime_minutes=uptime[2], uptime_seconds=uptime[3],
                recv=self.counter['recv'], send=self.counter['send'],
                error_count=self.counter['error'],
                cpu=get_cpu_name(), cpu_precent=psutil.cpu_percent(),
                memory_used=psutil.virtual_memory().used // 1048576,  # MiB
                memory_total=psutil.virtual_memory().total // 1048576,
                memory_precent=psutil.virtual_memory().percent,
                disk_used=disk_usage.used // 1.0737e+9,  # GiB
                disk_total=disk_usage.total // 1.0737e+9, disk_precent=disk_usage.percent
            ))

        return mirai_req

    async def _lolicon_api_common_request(self, mirai_res: MiraiResponse, keyword: str = None) -> MiraiRequest:
        """lolicon api请求

        :param mirai_res: Mirai响应对象
        :param keyword: 搜索关键词
        :return: MiraiRequest对象
        """
        # 创建MiraiRequest对象
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())
        mirai_req.from_resp(mirai_res)

        # 从api上拉取一张图片
        try:
            if keyword:
                res = httpx.get('https://api.lolicon.app/setu/v2?r18={r18}&keyword={keyword}'.format(
                    r18=app_config.commands.lolicon_api.allow_r18, keyword=keyword
                ))
            else:
                res = httpx.get('https://api.lolicon.app/setu/v2?r18={r18}'.format(
                    r18=app_config.commands.lolicon_api.allow_r18
                ))

            # 检查响应状态
            res.raise_for_status()

            # 检测可能的搜索错误
            if len(res.json()['data']) == 0:
                mirai_req.add_plain(app_lang.template.lolicon_api.api_search_error)
                return mirai_req

            # 检查缓存中是否已有同样的文件
            image_url = res.json()['data'][0]['urls']['original']
            image_filename = image_url.split('/')[-1]
            cache_path = f'cache/lolicon_api/{image_filename}'

            if not os.path.exists(cache_path):
                res_image = httpx.get(image_url)
                res_image.raise_for_status()

                with open(cache_path, 'wb') as f:
                    f.write(res_image.content)

        except httpx.HTTPError as e:
            logger.error(str(e))
            self.counter['error'] += 1

            if app_config.bot.send_exception:
                mirai_req.add_plain(str(e))
            return mirai_req

        # 处理水印
        if app_config.commands.lolicon_api.watermark.enable:
            logger.debug(app_lang.logs.lolicon_image_editing.format(img=image_filename))

            try:
                with Image.open(cache_path) as i:
                    font = ImageFont.truetype(
                        app_config.app.font,
                        app_config.commands.lolicon_api.watermark.size
                    )
                    drawer = ImageDraw.Draw(i)
                    drawer.text(
                        (0, 0),
                        text=app_config.commands.lolicon_api.watermark.format.format(
                            timestamp=time.time()
                        ),
                        font=font,
                        fill=(
                            app_config.commands.lolicon_api.watermark.color.r,
                            app_config.commands.lolicon_api.watermark.color.g,
                            app_config.commands.lolicon_api.watermark.color.b
                        )
                    )
                    modified_path = f'cache/lolicon_api/modified_{image_filename}'
                    i.save(modified_path)
                    cache_path = modified_path

            except UnidentifiedImageError as e:
                os.remove(cache_path)
                self.counter['error'] += 1
                logger.error(str(e))
                if app_config.bot.send_exception:
                    mirai_req.add_plain(str(e))
                return mirai_req

        # 读取图片并转换为base64
        with open(cache_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')

        # 处理标签
        _tags = ', '.join(res.json()['data'][0]['tags'])

        # 添加信息文本
        mirai_req.add_plain(app_lang.template.lolicon_api.info.format(
            title=res.json()['data'][0]['title'],
            author=res.json()['data'][0]['author'],
            pid=res.json()['data'][0]['pid'],
            tags=_tags,
            width=res.json()['data'][0]['width'],
            height=res.json()['data'][0]['height'],
            original_link=res.json()['data'][0]['urls']['original'] if
            app_config.commands.lolicon_api.send_original_link else 'None'
        ))

        # 添加图片
        mirai_req.add_image(
            _base64=image_b64,
            width=res.json()['data'][0]['width'],
            height=res.json()['data'][0]['height']
        )

        return mirai_req

    async def _command_lolicon_api(self, args: list, mirai_res: MiraiResponse) -> MiraiRequest:
        """处理lolicon API命令

        :param args: 命令参数列表
        :param mirai_res: Mirai响应对象
        :return: MiraiRequest对象
        """
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())
        mirai_req.from_resp(mirai_res)

        # 检测黑白名单
        # 白名单模式 (is_white_list = True):
        # - 用户必须在 user_list 中
        # - 群组必须在 group_list 中
        # - 任一条件不满足则拒绝访问
        # 黑名单模式 (is_white_list = False):
        # - 用户不能在 user_list 中
        # - 群组不能在 group_list 中
        # - 任一条件满足则拒绝访问
        is_user_in_list = mirai_res.is_in_user_list(app_config.commands.lolicon_api.user_list)
        is_group_in_list = mirai_res.is_in_group_list(app_config.commands.lolicon_api.group_list)
        if app_config.commands.lolicon_api.is_white_list:
            if not is_user_in_list or not is_group_in_list:
                mirai_req.add_plain(app_lang.template.lolicon_api.forbidden)
                logger.warning(app_lang.logs.command_lolicon_api.forbidden.format(
                    id=mirai_res.get_sender_id()
                ))
                return mirai_req
        else:  # 黑名单
            if is_user_in_list or is_group_in_list:
                mirai_req.add_plain(app_lang.template.lolicon_api.forbidden)
                logger.warning(app_lang.logs.command_lolicon_api.forbidden.format(
                    id=mirai_res.get_sender_id()
                ))
                return mirai_req

        # 处理不同命令
        if len(args) == 0:  # 默认行为
            return await self._lolicon_api_common_request(mirai_res)
        elif args[0] == 'test':
            mirai_req.add_plain(app_lang.template.lolicon_api.permission_ok)
            return mirai_req
        elif args[0] == 'help':
            mirai_req.add_plain(app_lang.template.lolicon_api.help.format(
                command_prefix=app_config.app.commands.prefix
            ))
            return mirai_req
        elif args[0] == 'search':
            if len(args) < 2:
                mirai_req.add_plain(app_lang.template.lolicon_api.too_less_arguments.format(
                    n=2, command_prefix=app_config.app.commands.prefix
                ))
                return mirai_req
            return await self._lolicon_api_common_request(mirai_res, args[1])
        else:
            mirai_req.add_plain(app_lang.template.lolicon_api.unknown_arguments.format(
                args=args[0], command_prefix=app_config.app.commands.prefix
            ))
            return mirai_req

    async def _command_dice(self, mirai_res: MiraiResponse) -> MiraiRequest:
        logger.debug(app_lang.logs.generate_dice_successful.format(id=mirai_res.get_sender_id()))
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())

        # 从mirai_res中加载目标对象
        mirai_req.from_resp(mirai_res)

        # 生成随机数
        # 我觉得可以修改范围（不过这是之后的事情了）
        random_number = random.randint(1, 6)

        mirai_req.add_plain(app_lang.template.dice.info.format(number=random_number))

        return mirai_req

    async def _command_openai_chat(self, args: list, mirai_res: MiraiResponse) -> MiraiRequest:
        # 创建MiraiRequest对象
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())
        mirai_req.from_resp(mirai_res)

        if len(args) == 0:
            mirai_req.add_plain(app_lang.template.openai_chat.too_less_arguments.format(
                n=1, command_prefix=app_config.app.commands.prefix
            ))
            return mirai_req
        elif args[0] == 'help':
            mirai_req.add_plain(app_lang.template.openai_chat.help.format(
                command_prefix=app_config.app.commands.prefix
            ))
            return mirai_req
        else:
            # 创建OpenAIChat对象
            openai_chat = OpenAIChat(
                mirai_res.get_group_id() if mirai_res.get_message_type() == 'GroupMessage' else mirai_res.get_sender_id(),
                is_group=(True if mirai_res.get_message_type() == 'GroupMessage' else False)
            )

            # 进入处理参数
            if args[0] == 'clean':
                if openai_chat.is_group:
                    logger.info(app_lang.logs.openai_chat.group_clean_chat.format(
                        id=mirai_res.get_sender_id(),
                        group_id=mirai_res.get_group_id()
                    ))
                else:
                    logger.info(app_lang.logs.openai_chat.user_clean_chat.format(
                        id=mirai_res.get_sender_id()
                    ))

                await openai_chat.create()
                await openai_chat.save_history()

                mirai_req.add_plain(app_lang.template.openai_chat.manual_clean_successful)
                return mirai_req
            elif args[0] == 'with':
                logger.info(
                    app_lang.logs.openai_chat.change_chat_present.format(
                        id=mirai_res.get_sender_id(), present=args[1]
                    )
                )
                try:
                    app_config.openai_chat.presents[args[1]]["conversations"]
                except KeyError:
                    mirai_req.add_plain(app_lang.template.openai_chat.present_not_exist.format(
                        present=args[1]
                    ))
                    return mirai_req
                else:
                    await openai_chat.create(
                        app_config.openai_chat.presents[args[1]]["conversations"]
                    )
                    await openai_chat.save_history()

                    mirai_req.add_plain(app_lang.template.openai_chat.use_present_successful.format(
                        present=args[1], present_display_name=app_config.openai_chat.presents[args[1]]
                        ["display_name"]
                    ))
                    return mirai_req
            else:
                mirai_req.add_plain(app_lang.template.openai_chat.unknown_arguments.format(
                    args=args[0]
                ))
                return mirai_req

    async def _command_wordcloud(self, args: list, mirai_res: MiraiResponse) -> MiraiRequest:
        # 创建MiraiRequest对象
        mirai_req = MiraiRequest(
            command='send' + mirai_res.get_message_type(),
        )
        mirai_req.from_resp(mirai_res)

        # 检查是否在非群聊中使用
        if mirai_res.get_message_type() != 'GroupMessage':
            mirai_req.add_plain(app_lang.template.wordcloud.not_in_group)
            return mirai_req

        if len(args) == 0:
            mirai_req.add_plain(app_lang.template.wordcloud.too_less_arguments.format(
                n=1, command_prefix=app_config.app.commands.prefix
            ))
            return mirai_req
        elif args[0] == 'help':
            mirai_req.add_plain(app_lang.template.wordcloud.help.format(
                command_prefix=app_config.app.commands.prefix
            ))
            return mirai_req
        elif args[0] == 'generate':
            wc = WordCloud(
                width=app_config.wordcloud.width,
                height=app_config.wordcloud.height,
                font_path=app_config.app.font,
                background_color=app_config.wordcloud.background_color,
                max_words=app_config.wordcloud.max_words,
                stopwords=app_config.wordcloud.stopwords,
                max_font_size=app_config.wordcloud.max_font_size,
                random_state=app_config.wordcloud.random_state,
                mask=app_config.wordcloud.mask,
                contour_color=app_config.wordcloud.contour_color,
                contour_width=app_config.wordcloud.contour_width,
                colormap=app_config.wordcloud.colormap,
                repeat=app_config.wordcloud.repeat,
            )

            # 打开保存的文件，分词后生成词云
            with open('data/wordcloud/%s.txt' % mirai_res.get_group_id(), 'r', encoding='utf-8') as f:
                wc.generate(' '.join(jieba.lcut(f.read())))

            # 保存词云
            wc.to_file('cache/wordcloud/wordcloud_%s.png' % mirai_res.get_group_id())

            # 发送词云
            with open('cache/wordcloud/wordcloud_%s.png' % mirai_res.get_group_id(), 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')

            mirai_req.add_image(_base64=image_b64)
            return mirai_req
        else:
            mirai_req.add_plain(app_lang.template.wordcloud.unknown_arguments.format(
                args=args[0], command_prefix=app_config.app.commands.prefix
            ))
            return mirai_req

    @on_message_types('GroupMessage', 'FriendMessage')
    @include_bot_at()
    @no_command_prefix()
    async def _openai_chat(self, mirai_res: MiraiResponse):
        """

        :param mirai_res: MiraiResponse
        :return: MiraiRequest
        """
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())
        mirai_req.from_resp(mirai_res)

        # 检查是否开启了群聊对话
        if mirai_res.get_message_type() == 'GroupMessage':
            if not app_config.openai_chat.group.enable:
                mirai_req.add_plain(app_lang.template.openai_chat.group_chat_disabled)
                return mirai_req

        # 创建对话对象
        openai_chat = OpenAIChat(
            mirai_res.get_group_id() if mirai_res.get_message_type() == 'GroupMessage' else mirai_res.get_sender_id(),
            is_group=(True if mirai_res.get_message_type() == 'GroupMessage' else False)
        )

        # 检查是否存在用户对话
        if await openai_chat.check_target_db_exist():
            await openai_chat.load_history()
        else:
            await openai_chat.create(
                present=app_config.openai_chat.presents.default.conversations,
            )

        # 检查是否加入用户名称到对话内容中
        if app_config.openai_chat.insert_username:
            # 添加对话内容
            await openai_chat.append(
                app_config.openai_chat.insert_username_format.format(
                    username=mirai_res.get_sender_name()[0],
                    original_text=mirai_res.__str__(use_strip=True, stringifies_at=False)
                ),
                "user",
            )
        else:
            await openai_chat.append(
                mirai_res.__str__(use_strip=True, stringifies_at=False),
                "user",
            )

        # 获取回复
        response = await openai_chat.generate()

        # 保存对话内容
        await openai_chat.save_history()

        # 发送消息
        mirai_req.add_plain(response)
        return await self.send_mirai_req(mirai_req)

    @on_message_types('GroupMessage')
    async def _bilibili_url_detect(self, mirai_res: MiraiResponse):
        # 检测黑白名单
        # 白名单模式 (is_white_list = True):
        # - 用户必须在 user_list 中
        # - 群组必须在 group_list 中
        # - 任一条件不满足则拒绝访问
        # 黑名单模式 (is_white_list = False):
        # - 用户不能在 user_list 中
        # - 群组不能在 group_list 中
        # - 任一条件满足则拒绝访问
        is_user_in_list = mirai_res.is_in_user_list(app_config.bilibili_url_detect.user_list)
        is_group_in_list = mirai_res.is_in_group_list(app_config.bilibili_url_detect.group_list)
        if app_config.bilibili_url_detect.is_white_list:
            if not is_user_in_list or not is_group_in_list:
                logger.warning(app_lang.logs.ignore_bilibili_video_detect.format(
                    id=mirai_res.get_sender_id(),
                    video_id="N/A"
                ))
                return None
        else:  # 黑名单
            if is_user_in_list or is_group_in_list:
                logger.warning(app_lang.logs.ignore_bilibili_video_detect.format(
                    id=mirai_res.get_sender_id(),
                    video_id="N/A"
                ))
                return None

        text = mirai_res.__str__()
        video_matches = list(re.finditer(
            app_config.bilibili_url_detect.regex.video['_'],
            text
        ))
        user_matches = list(re.finditer(
            app_config.bilibili_url_detect.regex.user['_'],
            text
        ))

        # 只创建一次 MiraiRequest
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())
        mirai_req.from_resp(mirai_res)

        # 先处理所有视频匹配
        for match in video_matches:
            video_id = match.group(app_config.bilibili_url_detect.regex.video.full_index)
            if video_id.lower().startswith('av'):
                video = bilibili_api.video.Video(
                    aid=int(video_id[2:]),
                    credential=self.bilibili_credential
                )
            else:
                video = bilibili_api.video.Video(
                    bvid=video_id,
                    credential=self.bilibili_credential
                )

            logger.debug(
                app_lang.logs.bilibili_video_match_successful.format(video_id=video_id)
            )

            try:
                video_info = await video.get_info()
            except bilibili_api.ResponseCodeException as e:
                logger.error(str(e))
                self.counter['error'] += 1
                if app_config.bot.send_exception:
                    mirai_req.add_plain(str(e))
                continue
            else:
                mirai_req.add_plain(app_lang.template.bilibili_url_detect.video.format(
                    title=video_info["title"],
                    owner_name=video_info["owner"]["name"],
                    desc=video_info['desc']
                ))
                mirai_req.add_image(url=video_info["pic"])

        # 再处理所有用户匹配
        for match in user_matches:
            user_id = match.group(app_config.bilibili_url_detect.regex.user.full_index)
            user = bilibili_api.user.User(
                user_id,
                credential=self.bilibili_credential
            )

            logger.debug(
                app_lang.logs.bilibili_user_url_match_successful.format(
                    content=user_id)
            )

            try:
                user_relation_info = await user.get_relation_info()
                user_info = await user.get_user_info()
            except bilibili_api.ResponseCodeException as e:
                logger.error(str(e))
                self.counter['error'] += 1
                if app_config.bot.send_exception:
                    mirai_req.add_plain(str(e))
                continue
            else:
                mirai_req.add_plain(app_lang.template.bilibili_url_detect.user.format(
                    name=user_info['name'],
                    sex=user_info['sex'],
                    birthday=user_info['birthday'],
                    follower=user_relation_info['follower'],
                    sign=user_info['sign']
                ))
                mirai_req.add_image(url=user_info["face"])

        # 只在有内容时发送
        if mirai_req.get_message_chain():
            await self.send_mirai_req(mirai_req)

        return None

    @on_message_types('GroupMessage')
    @no_command_prefix()  # 不需要命令前缀
    async def _wordcloud_collector(self, mirai_res: MiraiResponse):
        is_user_in_list = mirai_res.is_in_user_list(app_config.wordcloud.user_list)
        is_group_in_list = mirai_res.is_in_group_list(app_config.wordcloud.group_list)
        if app_config.wordcloud.is_white_list:
            if not is_user_in_list or not is_group_in_list:
                logger.debug(
                    app_lang.logs.wordcloud.ignore_wordcloud_collector.format(
                        id=mirai_res.get_group_id(),
                        content=mirai_res.__str__()
                    )
                )
                return None
        else:  # 黑名单
            if is_user_in_list or is_group_in_list:
                logger.debug(
                    app_lang.logs.wordcloud.ignore_wordcloud_collector.format(
                        id=mirai_res.get_group_id(),
                        content=mirai_res.__str__()
                    )
                )
                return None

        # 收集用户对话内容
        with open('data/wordcloud/%s.txt' % mirai_res.get_group_id(), 'a', encoding='utf-8') as f:
            f.write(mirai_res.__str__(use_strip=True, stringifies_at=False))
            f.write('\n')

        logger.debug(
            app_lang.logs.wordcloud.collected_wordcloud.format(
                id=mirai_res.get_group_id(),
                content=mirai_res.__str__()
            )
        )

    @on_message_types('GroupMessage', 'FriendMessage')
    async def auto_web_screenshot(self, mirai_res: MiraiResponse):
        # 准备Request对象
        mirai_req = MiraiRequest('send' + mirai_res.get_message_type())

        # 从mirai_res中加载目标对象
        mirai_req.from_resp(mirai_res)

        # 从mirai_res中提取url和站点域名
        url_match = re.search(
            r'https?://(?:'
            r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'  # 域名
            r'|'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'  # IPv4
            r'|'
            r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}'  # IPv6
            r'|'
            r'(?:[0-9a-fA-F]{1,4}:){1,7}:'  # IPv6 缩写
            r'|'
            r'(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}'  # IPv6 缩写
            r'|'
            r'(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}'  # IPv6 缩写
            r'|'
            r'(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}'  # IPv6 缩写
            r'|'
            r'(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}'  # IPv6 缩写
            r'|'
            r'(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}'  # IPv6 缩写
            r'|'
            r'[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}'  # IPv6 缩写
            r'|'
            r':(?::[0-9a-fA-F]{1,4}){1,7}'  # IPv6 缩写
            r'|'
            r'::'  # IPv6 缩写
            r')(?::\d+)?(?:/\S*)?',  # 端口号和路径
            mirai_res.__str__()
        )

        # 判断消息内是否有URL
        if not url_match:
            return mirai_req.cancel()

        # 初始化黑名单标志为False
        is_blacklisted = False

        is_user_in_list = mirai_res.is_in_user_list(app_config.auto_web_screenshot.user_list)
        is_group_in_list = mirai_res.is_in_group_list(app_config.auto_web_screenshot.group_list)
        if app_config.auto_web_screenshot.is_white_list:
            if not is_user_in_list or not is_group_in_list:
                is_blacklisted = True
        else:  # 黑名单
            if is_user_in_list or is_group_in_list:
                is_blacklisted = True

        # 提取完整URL
        full_url = url_match.group(0)
        domain = re.search(r'https?://([^/]+)', full_url).group(1)

        # 遍历黑名单中的所有模式
        for pattern in app_config.auto_web_screenshot.website_black_list:
            # 检查模式是否包含通配符(*)
            if '*' in pattern:
                # 将域名和模式都按点号分割成部分
                # 例如: "example.com" -> ["example", "com"]
                domain_parts = domain.split('.')
                pattern_parts = pattern.split('.')

                # 确保模式的部分数量不超过域名的部分数量
                # 例如: 不允许 "*.example.com" 匹配 "com"
                if len(pattern_parts) <= len(domain_parts):
                    # 从右向左进行匹配（因为域名是从右向左读的）
                    # 例如: "*.example.com" 应该匹配 "test.example.com"
                    matches = True
                    for i in range(len(pattern_parts)):
                        # 使用fnmatch进行通配符匹配
                        # 例如: "*.com" 可以匹配 "example.com"
                        if not fnmatch.fnmatch(domain_parts[-(i+1)], pattern_parts[-(i+1)]):
                            matches = False
                            break
                    # 如果所有部分都匹配，则标记为黑名单
                    if matches:
                        is_blacklisted = True
                        break
            # 处理精确匹配的情况（不包含通配符）
            # 例如: "example.com" 只匹配 "example.com"
            elif domain == pattern:
                is_blacklisted = True
                break

        if is_blacklisted:
            mode = app_config.auto_web_screenshot.website_black_list_mode
            if mode == 1:
                logger.security(app_lang.logs.web_screenshot.forbidden.format(url=full_url))
                return None # 静默拒绝
            elif mode == 2:
                logger.security(app_lang.logs.web_screenshot.forbidden.format(url=full_url))
                mirai_req.add_plain(app_lang.template.web_screenshot.forbidden)
                return await self.send_mirai_req(mirai_req)
            # mode == 0 允许访问，继续往下走

        # 开始截图
        # 创建WebScreenshotBrowser对象
        web_screenshot_browser = WebScreenshotBrowser(self.web_screenshot_service.get_driver())

        # 设置文件名称
        file_name = 'cache/web_screenshot/{domain}-{time}.png'.format(
            domain=domain,
            time=int(time.time())
        )

        try:
            await web_screenshot_browser.capture_full_page_screenshot(
                full_url, file_name,
                app_config.web_screenshot_common.implicitly_wait_time
            )
        except Exception as e:
            logger.error(str(e))
            if app_config.bot.send_exception:
                mirai_req.add_plain(str(e))
                return None
            return None
        else:
            with open(file_name, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
            mirai_req.add_image(_base64=image_b64)
        finally:
            # 关闭浏览器
            web_screenshot_browser.close()

            return await self.send_mirai_req(mirai_req)

    async def run_async(self):
        """异步开始运行机器人

        :return:
        """
        logger.info(app_lang.logs.start.format(version=APP_VERSION, platform=platform.platform()))
        self.startup_time = time.time()

        async for ws in websockets.connect(
                self.websocket_url
                + "?verifyKey="
                + self.verify_key
                + "&qq="
                + str(self.account)
        ):
            try:
                logger.info(
                    app_lang.logs.try_login.format(account=str(self.account))
                )

                # 更新Websocket对象
                self.ws = ws

                # 开始接收信息
                while True:
                    mirai_res = await self.recv()

                    # 运行待定计划任务
                    if app_config.schedule.enable:
                        await self.scheduler.async_run_pending()

                    # API返回代码查错
                    if mirai_res.get_response_code() == 0:  # API发送返回值/注册成功
                        # 如果session为空，尝试获取session
                        if not self.session:
                            self.session = mirai_res.get_session()
                            if not self.session:
                                raise RuntimeError  # 先这样，以后再改
                            else:
                                logger.info(
                                    app_lang.logs.get_session_successful.format(
                                        session=self.session
                                    )
                                )
                    elif mirai_res.get_response_code() is None:  # 事件消息
                        # 屏蔽机器人自己发出的消息
                        if mirai_res.get_sender_id() == app_config.bot.account:
                            logger.warning(app_lang.logs.ignore_bot_message.format(
                                content=mirai_res.__str__()
                            ))

                            # 直接跳过
                            continue

                        try:
                            # 模块：正则回复
                            # if app_config.regex.enable:
                            # pass

                            # 开始处理各种事件
                            await self._command_base(mirai_res)

                            # 模块：OpenAI聊天
                            if app_config.openai_chat.enable:
                                await self._openai_chat(mirai_res)

                            # 模块：B站URL检测
                            if app_config.bilibili_url_detect.enable:
                                await self._bilibili_url_detect(mirai_res)

                            # 模块：词云收集器
                            if app_config.wordcloud.enable:
                                await self._wordcloud_collector(mirai_res)

                            # 模块：网页截图
                            if app_config.auto_web_screenshot.enable:
                                await self.auto_web_screenshot(mirai_res)

                        except MessageCancelledException:  # 消息被取消
                            # logger.debug()
                            pass

                    elif mirai_res.get_response_code() == 1:
                        raise WrongVerifyKeyException(app_lang.logs.exceptions.wrong_verify_key_exception)
                    elif mirai_res.get_response_code() == 2:
                        raise BotNotExistException(app_lang.logs.exceptions.bot_not_exist_exceptions)
                    elif mirai_res.get_response_code() == 3:
                        raise SessionInvalidException
                    elif mirai_res.get_response_code() == 5:
                        raise TargetNotExistException
                    else:
                        raise RuntimeError
            except websockets.ConnectionClosed:
                logger.warning(app_lang.logs.try_reconnect)
                self.counter['error'] += 1
                continue

            except WrongVerifyKeyException:
                logger.critical(traceback.format_exc())

                return -1001
            except BotNotExistException:
                logger.critical(traceback.format_exc())

                return -1002
            except SessionInvalidException:
                logger.critical(traceback.format_exc())

                return -1003
            except KeyboardInterrupt:
                # 关闭WebScreenshotService
                if app_config.web_screenshot_common.enable:
                    self.web_screenshot_service.close()

                return 0
            except Exception:
                logger.critical(app_lang.logs.critical_exception)
                logger.critical(traceback.format_exc())
                self.counter['error'] += 1
                if app_config.app.auto_restart:
                    logger.warning(app_lang.logs.auto_restart_info)

                    # 重置时清空session与Websocket对象
                    self.session = None
                    self.ws = None
                    continue
                else:
                    return -1
        return -1


if __name__ == "__main__":
    # 初始化目录结构
    init_structure()

    # 获取配置
    app_config: Config = Config()

    # 压缩旧日志（如果有）
    compress_latest_logs(app_config.app.log_path)

    # 获取语言文件
    app_lang = Lang(app_config.app.lang)

    # 配置logging
    # 自定义日志级别
    logging.addLevelName(SECURITY_LOGGER_LEVEL, 'SECURITY')
    logging.setLoggerClass(CustomLogger)

    # 创建logger实例
    logger = logging.getLogger(__name__)
    logger.setLevel(app_config.app.log_level)

    # 配置文件处理器
    file_handler = logging.FileHandler(app_config.app.log_path)
    file_handler.setFormatter(logging.Formatter(app_config.app.log_format))
    logger.addHandler(file_handler)

    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(app_config.app.log_format))
    logger.addHandler(console_handler)

    # 创建机器人
    bot = Bot(
        websocket_url=app_config.bot.websocket_url,
        verify_key=app_config.bot.verify_key,
        account=app_config.bot.account,
    )

    # 启动机器人
    asyncio.run(bot.run_async())
