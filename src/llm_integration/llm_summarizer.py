# src/llm_integration/llm_summarizer.py (最终修正版)
import logging
import os
from typing import List
import dashscope
from dashscope.api_entities.dashscope_response import GenerationResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "qwen-plus"

class LLMSummarizationError(Exception):
    """自定义异常类，用于LLM总结相关的错误。"""
    pass

def _call_bailian_llm(prompt: str, model: str) -> str:
    """
    一个统一的函数，用于调用阿里云百炼LLM并处理响应和错误。
    """
    try:
        response: GenerationResponse = dashscope.Generation.call(
            model=model,
            prompt=prompt,
            result_format='text'
        )

        # 为调试添加日志，显示完整的API响应
        logger.info(f"成功收到百炼API响应: {response}")

        if response.status_code == 200:
            # 核心修正：从 response.output 字典中提取 'text' 键的值
            return response.output['text'].strip()
        else:
            err_msg = f"阿里云百炼API请求失败。状态码: {response.status_code}, 错误信息: {response.message}"
            logger.error(err_msg)
            raise LLMSummarizationError(err_msg)
            
    except Exception as e:
        logger.error(f"调用阿里云百炼API时发生未知错误: {e}", exc_info=True)
        raise LLMSummarizationError(f"调用LLM失败: {e}")

def llm_assisted_text_cleaning(raw_ocr_text: str, **kwargs) -> str:
    """
    利用阿里云百炼LLM对OCR文本进行校正和清洗。
    """
    if not isinstance(raw_ocr_text, str):
        raise LLMSummarizationError("输入 raw_ocr_text 必须是字符串。")
    if not os.getenv("DASHSCOPE_API_KEY"):
         raise LLMSummarizationError("环境变量 DASHSCOPE_API_KEY 未设置。")

    prompt = f"""
    请作为一名文本清洗专家，仔细阅读以下OCR识别出的原始文本。
    你的任务是：纠正明显的OCR错误，移除所有非聊天内容的噪音（如时间戳、日期、[图片]、[语音]、[表情]等），恢复原始对话的流畅性和可读性。不要添加任何总结或评论，只返回清洗后的原始对话内容。

    原始文本：
    {raw_ocr_text}
    """
    return _call_bailian_llm(prompt, DEFAULT_LLM_MODEL)

def summarize_chat_history(chat_messages: List[str], **kwargs) -> str:
    """
    使用阿里云百炼LLM总结聊天记录。
    """
    if not isinstance(chat_messages, list) or not all(isinstance(msg, str) for msg in chat_messages):
        raise LLMSummarizationError("输入 chat_messages 必须是字符串列表。")
    if not os.getenv("DASHSCOPE_API_KEY"):
         raise LLMSummarizationError("环境变量 DASHSCOPE_API_KEY 未设置。")

    full_chat_text = "\n".join(chat_messages)
    
    logger.info("聊天记录长度适中，直接进行总结。")
    
    prompt = f"""
    请总结以下微信聊天记录的关键信息、决策点和待办事项。请以结构化的方式呈现，例如使用要点、列表或简明段落。

    聊天记录：
    {full_chat_text}
    """
    return _call_bailian_llm(prompt, DEFAULT_LLM_MODEL)