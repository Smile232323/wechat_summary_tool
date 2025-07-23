# src/ocr_processing/ocr_processor.py
import pytesseract
import logging
import re
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import concurrent.futures # 用于并行处理

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Tesseract OCR Engine 路径配置 (全局默认设置) ---
# 你的 tesseract.exe 文件的实际路径。
# 即使Tesseract已添加到系统PATH中，明确设置此行可以增加程序的鲁棒性。
# 如果你希望完全依赖系统PATH，可以注释掉此行。
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # <--- 放在这里！

# ------------------------------------

# 发言人识别常量
SPEAKER_ME = "我"
SPEAKER_OTHER = "对方"

class OCRPreprocessingError(Exception):
    """自定义异常类，用于OCR和预处理相关的错误。"""
    pass

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    目的：对图像进行预处理，以提高OCR识别率。
    输入：PIL.Image.Image 对象（原始截图）。
    输出：PIL.Image.Image 对象（预处理后的图像）。
    实现建议：灰度化、二值化、调整对比度、锐化、去噪等。
    """
    if not isinstance(image, Image.Image):
        logger.error("无效输入：image 必须是 PIL.Image.Image 对象。")
        raise OCRPreprocessingError("输入必须是 PIL.Image.Image 对象。")

    try:
        # 1. 转换为灰度图
        img_gray = image.convert('L')
        logger.debug("图像已转换为灰度图。")

        # 2. 调整对比度 (可选，可能改善某些图像的识别效果)
        enhancer = ImageEnhance.Contrast(img_gray)
        img_contrast = enhancer.enhance(1.5) # 增加50%对比度
        logger.debug("图像对比度已增强。")

        # 3. 锐化 (可选，可能改善某些图像的识别效果)
        img_sharpened = img_contrast.filter(ImageFilter.SHARPEN)
        logger.debug("图像已锐化。")

        # 4. 将PIL图像转换为OpenCV格式，以便进行更高级的处理（如二值化、降噪）
        img_np = np.array(img_sharpened)

        # 应用自适应阈值处理（比简单的大津法更适合光照不均的截图）
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C 或 cv2.ADAPTIVE_THRESH_MEAN_C
        # cv2.THRESH_BINARY 或 cv2.THRESH_BINARY_INV
        img_binarized = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        logger.debug("图像已使用自适应阈值进行二值化。")

        # 5. 降噪 (中值滤波对椒盐噪声效果好，截图常见此类噪声)
        img_denoised = cv2.medianBlur(img_binarized, 3) # 3x3 核大小
        logger.debug("图像已使用中值滤波进行降噪。")

        # 转换回PIL图像
        processed_image = Image.fromarray(img_denoised)
        logger.info("图像预处理完成。")
        return processed_image
    except Exception as e:
        logger.error(f"图像预处理过程中发生错误: {e}", exc_info=True)
        raise OCRPreprocessingError(f"未能预处理图像: {e}")

def perform_ocr(image: Image.Image, ocr_config: Dict) -> Tuple[str, float]:
    """
    目的：对图像执行OCR识别，并返回识别文本和置信度。
    输入：
    image: PIL.Image.Image 对象（预处理后的图像）。
    ocr_config: 包含OCR引擎偏好（例如：'local' 或 'cloud'）、API Key等配置的字典。
    输出：一个元组 (recognized_text, confidence)，其中 recognized_text 是字符串，confidence 是浮点数（0-100）。
    混合策略：内部应包含优先使用本地Tesseract，当置信度低于阈值或指定时，回退到云端OCR（例如Google Cloud Vision）。
    请为云端OCR的调用预留接口，但不需要实际调用，只需模拟其行为或提供占位符。
    """
    if not isinstance(image, Image.Image):
        logger.error("无效输入：image 必须是 PIL.Image.Image 对象。")
        raise OCRPreprocessingError("输入必须是 PIL.Image.Image 对象。")
    if not isinstance(ocr_config, dict):
        logger.error("无效输入：ocr_config 必须是字典。")
        raise OCRPreprocessingError("输入 ocr_config 必须是字典。")

    recognized_text = ""
    confidence = 0.0
    
    # --- 配置 Tesseract 路径 (如果 ocr_config 中提供，将覆盖全局设置) ---
    tesseract_cmd_from_config = ocr_config.get('tesseract_cmd')
    if tesseract_cmd_from_config:
        if os.path.exists(tesseract_cmd_from_config):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_from_config
            logger.debug(f"Pytesseract 命令路径已设置为: {tesseract_cmd_from_config} (来自配置)")
        else:
            logger.warning(f"配置中指定的 Tesseract 命令路径 '{tesseract_cmd_from_config}' 未找到。将尝试使用全局设置或系统PATH中的Tesseract。")
    else:
        logger.debug("未在配置中指定 Tesseract 命令路径，将使用全局设置或系统PATH中的Tesseract。")
    # ------------------------------------------------------------------

    local_ocr_confidence_threshold = ocr_config.get('local_ocr_confidence_threshold', 70.0)
    force_cloud_ocr = ocr_config.get('force_cloud_ocr', False)
    
    # --- 本地 OCR (Tesseract) ---
    try:
        logger.info("尝试使用 Tesseract 进行本地 OCR...")
        # pytesseract.image_to_data 提供每个单词/行的置信度
        # 对于整体置信度，我们可以平均单词置信度。
        # 使用 lang='chi_sim+eng' 确保同时识别中文和英文
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='chi_sim+eng')
        
        text_list = []
        confidences = []
        
        for i in range(len(data['text'])):
            # -1 表示非文本元素，跳过；确保文本非空
            if int(data['conf'][i]) > -1 and data['text'][i].strip(): 
                text_list.append(data['text'][i])
                confidences.append(int(data['conf'][i]))
        
        recognized_text = " ".join(text_list).strip()
        if confidences:
            confidence = sum(confidences) / len(confidences)
        else:
            confidence = 0.0
        
        logger.info(f"本地 OCR 完成。置信度: {confidence:.2f}")

        # 判断是否需要回退到云端 OCR
        if not force_cloud_ocr and confidence >= local_ocr_confidence_threshold:
            logger.info("本地 OCR 置信度足够。跳过云端 OCR。")
            return recognized_text, confidence
        else:
            if force_cloud_ocr:
                logger.info("根据配置强制使用云端 OCR。")
            else:
                logger.info(f"本地 OCR 置信度 ({confidence:.2f}) 低于阈值 ({local_ocr_confidence_threshold:.2f})。回退到云端 OCR。")

    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract 未安装或未在 PATH 中找到。请安装 Tesseract-OCR 或在 ocr_config 中提供其路径。", exc_info=True)
        # 如果 Tesseract 未找到，如果启用了云端 OCR 则必须回退，否则抛出错误。
        if not ocr_config.get('cloud_ocr_enabled', False):
            raise OCRPreprocessingError("Tesseract 未找到且未启用云端 OCR。无法执行 OCR。")
        logger.warning("由于 Tesseract 未找到，回退到云端 OCR。")
    except Exception as e:
        logger.error(f"本地 OCR 过程中发生错误: {e}", exc_info=True)
        if not ocr_config.get('cloud_ocr_enabled', False):
            raise OCRPreprocessingError(f"本地 OCR 失败且未启用云端 OCR: {e}")
        logger.warning("由于本地 OCR 错误，回退到云端 OCR。")

    # --- 云端 OCR 占位符 ---
    if ocr_config.get('cloud_ocr_enabled', False):
        logger.info("尝试云端 OCR (占位符)...")
        # 模拟云端 OCR 行为
        cloud_api_key = ocr_config.get('cloud_ocr_api_key')
        if not cloud_api_key:
            logger.warning("已启用云端 OCR，但 ocr_config 中缺少 'cloud_ocr_api_key'。")
            # 如果本地 OCR 有结果，作为最后手段使用；否则抛出错误。
            if recognized_text: 
                logger.warning("由于缺少云端 OCR API 密钥，将使用本地 OCR 结果。")
                return recognized_text, confidence
            else:
                raise OCRPreprocessingError("已启用云端 OCR 但无 API 密钥，且本地 OCR 失败。")

        # --- 模拟云端 OCR 调用 ---
        # 在实际场景中，您会调用 Google Cloud Vision API、AWS Textract 等。
        # 对于此任务，我们仅模拟一个结果。
        mock_cloud_text = "这是模拟的云端 OCR 结果。 " + recognized_text # 在本地结果基础上添加模拟前缀
        mock_cloud_confidence = 95.0 # 假设云端置信度较高
        logger.info("云端 OCR (占位符) 完成。")
        return mock_cloud_text, mock_cloud_confidence
    else:
        logger.warning("ocr_config 中未启用云端 OCR。将返回本地 OCR 结果（即使置信度低）或空（如果失败）。")
        if not recognized_text:
            raise OCRPreprocessingError("未从本地或云端（未启用）获取到 OCR 结果。")
        return recognized_text, confidence # 如果云端未启用，返回本地 OCR 结果

def clean_and_structure_chat_text(raw_text: str, image: Image.Image) -> List[str]:
    """
    目的：对OCR识别出的原始文本进行智能清洗，去除噪音，并识别发言人，最终输出结构化的聊天记录列表。
    输入：
    raw_text: OCR引擎返回的原始文本字符串 (此参数在当前实现中主要用于日志，实际内容解析依赖于image)。
    image: 对应的原始截图（用于发言人识别，例如通过分析聊天气泡X坐标）。
    输出：List[str]，其中每个字符串代表一条聊天消息，格式为 "{发言人}: {消息内容}"（例如："我: 你好", "对方: 很高兴认识你"）。
    清洗建议：使用正则表达式去除时间戳、日期、[图片]、[语音]、[表情]等。
    发言人识别：分析image中聊天气泡的像素位置（例如，左侧为对方，右侧为自己）来判断发言人。
    """
    if not isinstance(raw_text, str):
        logger.error("无效输入：raw_text 必须是字符串。")
        raise OCRPreprocessingError("输入 raw_text 必须是字符串。")
    if not isinstance(image, Image.Image):
        logger.error("无效输入：image 必须是 PIL.Image.Image 对象。")
        raise OCRPreprocessingError("输入 image 必须是 PIL.Image.Image 对象。")

    structured_messages: List[str] = []

    try:
        # 步骤 1: 获取详细的 OCR 数据 (边界框) 以进行发言人识别
        # 将 PIL 图像转换为 OpenCV 格式，用于 pytesseract.image_to_data
        img_np = np.array(image.convert('L')) # 确保灰度图以进行一致处理
        
        # 使用 pytesseract.image_to_data 获取单词级别的边界框和文本
        # 这对于基于 X 坐标的发言人识别至关重要
        data = pytesseract.image_to_data(img_np, output_type=pytesseract.Output.DICT, lang='chi_sim+eng')

        # 按发言人分组消息
        current_speaker: Optional[str] = None
        current_message_lines: List[str] = []
        
        image_width = image.width
        # 假设屏幕中间线是区分左右发言人的依据
        mid_x = image_width / 2 

        # 遍历 Tesseract 识别出的每个单词/文本块
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # 跳过空文本或置信度非常低的单词
            if not text or conf < 60: # 对单个单词使用置信度阈值
                continue

            # 获取单词的边界框信息
            x, _, _, _ = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # 根据单词边界框的 X 坐标判断发言人
            # 假设左半部分是“对方”，右半部分是“我”。
            speaker = SPEAKER_OTHER if x < mid_x else SPEAKER_ME
            
            # 如果发言人改变或这是第一个单词，则开始一条新消息
            if speaker != current_speaker:
                if current_message_lines:
                    # 清理并添加上一条消息
                    full_message = " ".join(current_message_lines).strip()
                    # 传递 current_speaker 给 _clean_chat_message 以处理冗余发言人名称
                    cleaned_message = _clean_chat_message(full_message, current_speaker) # <--- 修正点
                    if cleaned_message: # 清理后非空才添加
                        structured_messages.append(f"{current_speaker}: {cleaned_message}")
                
                current_speaker = speaker
                current_message_lines = [text]
            else:
                # 添加到当前消息
                current_message_lines.append(text)
        
        # 添加最后一条消息 (如果有的话)
        if current_message_lines:
            full_message = " ".join(current_message_lines).strip()
            # 传递 current_speaker 给 _clean_chat_message 以处理冗余发言人名称
            cleaned_message = _clean_chat_message(full_message, current_speaker) # <--- 修正点
            if cleaned_message:
                structured_messages.append(f"{current_speaker}: {cleaned_message}")

        logger.info(f"聊天文本清洗和结构化完成。发现 {len(structured_messages)} 条消息。")
        return structured_messages

    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract 未安装或未在 PATH 中找到。无法通过图像数据进行发言人识别。", exc_info=True)
        # 如果 Tesseract 不可用，则回退到不带发言人识别的基础文本清洗
        logger.warning("由于 Tesseract 错误，回退到不带发言人识别的基础文本清洗。")
        cleaned_text = _clean_chat_message(raw_text) # <--- 修正点，这里不传 speaker_name
        if cleaned_text:
            return [f"未知发言人: {cleaned_text}"]
        return []
    except Exception as e:
        logger.error(f"聊天文本清洗和结构化过程中发生错误: {e}", exc_info=True)
        raise OCRPreprocessingError(f"未能清洗和结构化聊天文本: {e}")

def _clean_chat_message(message: str, speaker_name: Optional[str] = None) -> str: # <--- 修正点：添加 speaker_name 参数
    """
    辅助函数，用于清洗单个聊天消息字符串。
    去除常见的微信伪影，如时间戳、日期、[图片]、[语音]、[表情]等。
    如果提供了发言人名称，也会尝试去除消息开头冗余的发言人名称。
    """
    # 去除时间戳 (例如："上午 10:30", "2023年10月26日 10:30")
    # 匹配常见的时间格式：HH:MM, HH:MM:SS, 上午/下午/凌晨/晚上 HH:MM，以及带年/月/日的日期
    message = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b|\b(上午|下午|凌晨|晚上)\s*\d{1,2}:\d{2}\b', '', message)
    message = re.sub(r'\b\d{4}年\d{1,2}月\d{1,2}日\b', '', message)
    message = re.sub(r'\b\d{1,2}月\d{1,2}日\b', '', message) # 例如："10月26日"

    # 去除常见的微信占位符
    message = re.sub(r'\[图片\]|\[语音\]|\[表情\]|\[文件\]|\[视频\]|\[链接\]', '', message)
    
    # --- 修正点：去除消息开头冗余的发言人名称 ---
    if speaker_name:
        # 构建正则表达式，匹配发言人名称在消息开头，后面跟着冒号、空格或没有分隔符
        # 使用 re.escape 来处理 speaker_name 中可能存在的特殊字符
        # \s* 匹配0个或多个空格，[:：\s]* 匹配0个或多个半角冒号、全角冒号或空格
        pattern = r"^\s*" + re.escape(speaker_name) + r"[:：\s]*" # <--- 修正点：添加全角冒号
        message = re.sub(pattern, "", message, 1, re.IGNORECASE) # 只替换一次，不区分大小写
    # ----------------------------------------------------

    # 去除多余的空白字符，并去除首尾空格
    message = re.sub(r'\s+', ' ', message).strip()
    
    return message

# 依赖列表 (供人类协调者参考，用于生成 requirements.txt)
# Pillow>=9.0.0
# pytesseract>=0.3.10
# opencv-python>=4.5.0
# numpy>=1.20.0
# concurrent.futures (Python标准库，无需pip安装)