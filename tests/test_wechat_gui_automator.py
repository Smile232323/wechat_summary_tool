import logging
from typing import List, Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, BlockedPromptException
from google.api_core.exceptions import GoogleAPIError

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认LLM模型名称
DEFAULT_LLM_MODEL = "gemini-pro" # 或者 "gemini-1.5-flash-latest" 或其他你希望使用的模型

class LLMSummarizationError(Exception):
    """自定义异常类，用于LLM总结相关的错误。"""
    pass

def llm_assisted_text_cleaning(raw_ocr_text: str, llm_api_key: str) -> str:
    """
    目的：利用LLM的上下文理解能力，对OCR后的原始文本进行校正和清洗，尤其是在OCR错误较多时。
    输入：
    raw_ocr_text: 从OCR模块接收到的原始或初步清洗的文本。
    llm_api_key: LLM服务的API Key。此Key应由调用方提供，不应硬编码。
    输出：str（经过LLM校正和清洗后的文本）。
    安全性与隐私：明确此操作涉及将用户数据发送到第三方LLM服务。在代码中应有注释说明。
    错误处理：任何API调用失败应使用 logging.error 记录，并抛出有意义的异常。
    """
    if not isinstance(raw_ocr_text, str):
        logger.error("无效输入：raw_ocr_text 必须是字符串。")
        raise LLMSummarizationError("输入 raw_ocr_text 必须是字符串。")
    if not llm_api_key:
        logger.error("LLM API Key 不能为空。")
        raise LLMSummarizationError("LLM API Key 缺失。")

    # 安全性与隐私：用户数据将发送到Google Gemini服务
    # 用户应知晓并同意此数据处理。
    genai.configure(api_key=llm_api_key)

    try:
        model = genai.GenerativeModel(DEFAULT_LLM_MODEL)
        prompt = f"""
        请作为一名文本清洗专家，仔细阅读以下OCR识别出的原始文本。
        这段文本可能包含OCR错误、乱码、多余的符号、时间戳、日期、[图片]、[语音]、[表情]等非聊天内容。
        你的任务是：
        1. 纠正明显的OCR错误。
        2. 移除所有非聊天内容的噪音（如时间戳、日期、[图片]、[语音]、[表情]等）。
        3. 恢复原始对话的流畅性和可读性。
        4. 不要添加任何总结或评论，只返回清洗后的原始对话内容。

        原始文本：
        {raw_ocr_text}
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except BlockedPromptException as e:
        logger.error(f"LLM辅助文本清洗被阻止：{e}", exc_info=True)
        raise LLMSummarizationError(f"LLM辅助文本清洗被阻止：{e}")
    except GoogleAPIError as e:
        logger.error(f"LLM辅助文本清洗API错误: {e}", exc_info=True)
        raise LLMSummarizationError(f"LLM辅助文本清洗API错误: {e}")
    except Exception as e:
        logger.error(f"LLM辅助文本清洗过程中发生未知错误: {e}", exc_info=True)
        raise LLMSummarizationError(f"LLM辅助文本清洗失败: {e}")

def summarize_chat_history(chat_messages: List[str], llm_api_key: str, max_tokens: int = 4000) -> str:
    """
    目的：将结构化的聊天记录列表发送给LLM，并获取其总结。
    输入：
    chat_messages: 从OCR/文本预处理模块接收到的 List[str] 格式的聊天记录。
    llm_api_key: LLM服务的API Key。此Key应由调用方提供，不应硬编码。
    max_tokens: LLM上下文窗口的限制（用于内部处理长文本）。
    输出：str（LLM生成的总结结果）。
    安全性与隐私：明确此操作涉及将用户数据发送到第三方LLM服务。在代码中应有注释说明。
    错误处理：任何API调用失败应使用 logging.error 记录，并抛出有意义的异常。
    """
    if not isinstance(chat_messages, list) or not all(isinstance(msg, str) for msg in chat_messages):
        logger.error("无效输入：chat_messages 必须是字符串列表。")
        raise LLMSummarizationError("输入 chat_messages 必须是字符串列表。")
    if not llm_api_key:
        logger.error("LLM API Key 不能为空。")
        raise LLMSummarizationError("LLM API Key 缺失。")

    # 安全性与隐私：用户数据将发送到Google Gemini服务
    # 用户应知晓并同意此数据处理。
    genai.configure(api_key=llm_api_key)

    full_chat_text = "\n".join(chat_messages)
    
    # 估算文本长度，考虑LLM的上下文窗口限制
    # 这是一个粗略的估算，实际token数可能不同
    estimated_tokens = len(full_chat_text) // 4 # 假设每个token约4个字符

    summary_prompt_template = """
    请总结以下微信聊天记录的关键信息、决策点和待办事项。请以结构化的方式呈现，例如使用要点、列表或简明段落。
    如果聊天记录很长，请分段总结并最终整合。

    聊天记录：
    {}
    """
    
    # 简单分段逻辑 (如果聊天记录非常长)
    # 实际的token计算会更复杂，这里用字符长度粗略模拟
    # 调整 max_tokens 默认值，使其更接近实际模型限制，并确保测试中的分段逻辑能触发
    if estimated_tokens > max_tokens * 0.7: # 降低阈值，更容易触发分段
        logger.info(f"聊天记录过长 ({estimated_tokens} tokens)，将尝试分段总结。")
        segments = []
        current_segment_lines = []
        current_segment_length = 0
        
        segment_max_chars = int(max_tokens * 0.6 * 4) # 假设1 token = 4 chars, 留更多余量

        for message in chat_messages:
            message_length = len(message)
            # 如果当前段落加上新消息会超过最大字符限制，并且当前段落不为空，则开始新段落
            if (current_segment_length + message_length > segment_max_chars) and current_segment_lines:
                segments.append("\n".join(current_segment_lines))
                current_segment_lines = []
                current_segment_length = 0
            
            current_segment_lines.append(message)
            current_segment_length += message_length
        
        if current_segment_lines:
            segments.append("\n".join(current_segment_lines))

        intermediate_summaries = []
        for i, segment in enumerate(segments):
            logger.info(f"总结第 {i+1}/{len(segments)} 段聊天记录...")
            segment_prompt = summary_prompt_template.format(segment)
            try:
                response = genai.GenerativeModel(DEFAULT_LLM_MODEL).generate_content(segment_prompt)
                intermediate_summaries.append(response.text.strip())
            except BlockedPromptException as e:
                logger.error(f"第 {i+1} 段总结被阻止：{e}", exc_info=True)
                raise LLMSummarizationError(f"第 {i+1} 段总结被阻止：{e}")
            except GoogleAPIError as e:
                logger.error(f"第 {i+1} 段总结API错误: {e}", exc_info=True)
                raise LLMSummarizationError(f"第 {i+1} 段总结API错误: {e}")
            except Exception as e:
                logger.error(f"第 {i+1} 段总结过程中发生未知错误: {e}", exc_info=True)
                raise LLMSummarizationError(f"第 {i+1} 段总结失败: {e}")

        if len(intermediate_summaries) > 1:
            logger.info("进行最终总结...")
            final_summary_prompt = "请整合以下分段总结，形成一个连贯、全面的最终总结：\n\n" + "\n\n".join(intermediate_summaries)
            try:
                response = genai.GenerativeModel(DEFAULT_LLM_MODEL).generate_content(final_summary_prompt)
                return response.text.strip()
            except BlockedPromptException as e:
                logger.error(f"最终总结被阻止：{e}", exc_info=True)
                raise LLMSummarizationError(f"最终总结被阻止：{e}")
            except GoogleAPIError as e:
                logger.error(f"最终总结API错误: {e}", exc_info=True)
                raise LLMSummarizationError(f"最终总结API错误: {e}")
            except Exception as e:
                logger.error(f"最终总结过程中发生未知错误: {e}", exc_info=True)
                raise LLMSummarizationError(f"最终总结失败: {e}")
        else:
            return intermediate_summaries[0] if intermediate_summaries else ""
    else:
        logger.info("聊天记录长度适中，直接进行总结。")
        try:
            prompt = summary_prompt_template.format(full_chat_text)
            model = genai.GenerativeModel(DEFAULT_LLM_MODEL)
            response = model.generate_content(prompt)
            return response.text.strip()
        except BlockedPromptException as e:
            logger.error(f"直接总结被阻止：{e}", exc_info=True)
            raise LLMSummarizationError(f"直接总结被阻止：{e}")
        except GoogleAPIError as e:
            logger.error(f"直接总结API错误: {e}", exc_info=True)
            raise LLMSummarizationError(f"直接总结API错误: {e}")
        except Exception as e:
            logger.error(f"直接总结过程中发生未知错误: {e}", exc_info=True)
            raise LLMSummarizationError(f"直接总结失败: {e}")

# 依赖列表 (供人类协调者参考，用于生成 requirements.txt)
# google-generativeai>=0.3.0 # 确保版本兼容
# google-api-core # 确保版本兼容