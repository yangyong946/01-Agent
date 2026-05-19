# config.py
import os
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")