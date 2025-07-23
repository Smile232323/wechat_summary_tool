# temp_test_interfaces.py (最终修正版，已修复Logger定义)
import os
import time
import logging # <-- 已导入
from dotenv import load_dotenv
from PIL import Image
from difflib import SequenceMatcher

load_dotenv()

from src.gui_automation.wechat_gui_automator import capture_chat_history_dynamically, GUIAutomationError
from src.ocr_processing.ocr_processor import clean_and_structure_chat_text, OCRPreprocessingError 
from src.llm_integration.llm_summarizer import summarize_chat_history, LLMSummarizationError

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # <--- 已添加这行缺失的代码

LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY")

TEST_REGION = (85, 120, 740, 900) 
MAX_SCROLLS = 15
SCREENSHOT_DIR = "temp_screenshots"

def deduplicate_messages_smartly(messages: list, similarity_threshold: float = 0.9) -> list:
    """
    使用字符串相似度来去除重复的消息。
    """
    if not messages:
        return []

    logger.info(f"开始智能去重，原始消息数: {len(messages)}，相似度阈值: {similarity_threshold}")
    unique_messages = [messages[0]]

    for i in range(1, len(messages)):
        current_message = messages[i]
        last_unique_message = unique_messages[-1]
        
        ratio = SequenceMatcher(None, last_unique_message, current_message).ratio()
        
        if ratio < similarity_threshold:
            unique_messages.append(current_message)
        else:
            # 现在 logger.info 可以正常工作了
            logger.info(f"跳过相似度为 {ratio:.2f} 的重复消息: '{current_message}' (与 '{last_unique_message}' 相似)")

    logger.info(f"智能去重完成，最终消息数: {len(unique_messages)}")
    return unique_messages

def main():
    if not LLM_API_KEY:
        logger.error("错误：环境变量 DASHSCOPE_API_KEY 未设置。")
        return

    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)

    all_chat_lines = []
    try:
        logger.info("步骤 1/3: 开始智能自动化截图...")
        logger.info(">>> 请在5秒内，将鼠标光标移动到微信聊天记录区域内 <<<")
        time.sleep(5)

        image_paths = capture_chat_history_dynamically(
            region=TEST_REGION,
            screenshot_dir=SCREENSHOT_DIR,
            max_scrolls=MAX_SCROLLS
        )

        if not image_paths:
            logger.warning("未能捕获任何截图，程序终止。")
            return
            
        logger.info(f"成功捕获 {len(image_paths)} 张截图。")

        logger.info("步骤 2/3: 开始OCR处理...")
        for i, path in enumerate(image_paths):
            logger.info(f"  - 正在处理图片 {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            with Image.open(path) as img:
                structured_lines = clean_and_structure_chat_text(raw_text="", image=img)
                all_chat_lines.extend(structured_lines)
        
        unique_lines = deduplicate_messages_smartly(all_chat_lines)
        
        logger.info(f"OCR与去重处理完成，共提取到 {len(unique_lines)} 条独立对话。")

        logger.info("步骤 3/3: 发送至阿里云百炼进行总结...")
        if not unique_lines:
            logger.warning("没有提取到任何聊天内容，无法进行总结。")
            return

        summary = summarize_chat_history(unique_lines)
        
        print("\n" + "="*20 + " 最 终 总 结 " + "="*20)
        print(summary)
        print("="*55 + "\n")

    except (GUIAutomationError, OCRPreprocessingError, LLMSummarizationError) as e:
        logger.error(f"处理流程中发生错误: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"发生未知异常: {e}", exc_info=True)
    finally:
        logger.info("集成测试脚本执行完毕。")

if __name__ == "__main__":
    main()