import unittest
from unittest.mock import patch, MagicMock
import os
import io
from PIL import Image
import numpy as np
import logging
import pytesseract # <-- 确保这一行存在

# --- 关键修正：确保能正确导入 src/ocr_processing/ocr_processor ---
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ocr_processing import ocr_processor
# ------------------------------------------------------------------

# 辅助函数：创建虚拟图像
def create_dummy_image(width=300, height=100, color=(255, 255, 255)):
    """创建一个指定宽度、高度和颜色的PIL图像。"""
    return Image.new('RGB', (width, height), color)

class TestOCRPreprocessingModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试运行前执行一次，用于设置全局配置，例如抑制日志输出。"""
        # 在测试期间抑制日志输出，以获得更清晰的测试报告
        logging.getLogger('src.ocr_processing.ocr_processor').setLevel(logging.CRITICAL)
        logging.getLogger('PIL').setLevel(logging.CRITICAL) # 抑制 PIL 警告
        logging.getLogger('pytesseract').setLevel(logging.CRITICAL) # 抑制 pytesseract 警告

    @classmethod
    def tearDownClass(cls):
        """在所有测试运行后执行一次，用于清理全局配置，例如恢复日志级别。"""
        # 恢复日志级别
        logging.getLogger('src.ocr_processing.ocr_processor').setLevel(logging.INFO)
        logging.getLogger('PIL').setLevel(logging.INFO)
        logging.getLogger('pytesseract').setLevel(logging.INFO)

    def setUp(self):
        """在每个测试方法运行前执行。"""
        self.dummy_image = create_dummy_image()
        self.sample_image_path = "test_sample_image.png"
        # 创建一个简单的图像文件，用于测试文件操作（如果需要，尽管当前模块不直接处理文件）
        self.dummy_image.save(self.sample_image_path)

    def tearDown(self):
        """在每个测试方法运行后执行。"""
        # 清理创建的测试文件
        if os.path.exists(self.sample_image_path):
            os.remove(self.sample_image_path)

    # --- test_preprocess_image_for_ocr ---
    def test_preprocess_image_for_ocr_valid_input(self):
        """测试有效PIL图像输入的图像预处理。"""
        processed_image = ocr_processor.preprocess_image_for_ocr(self.dummy_image)
        self.assertIsInstance(processed_image, Image.Image)
        self.assertEqual(processed_image.mode, 'L') # 预处理后应为灰度图
        self.assertEqual(processed_image.size, self.dummy_image.size)

    def test_preprocess_image_for_ocr_invalid_input(self):
        """测试无效输入的图像预处理，应抛出 OCRPreprocessingError。"""
        with self.assertRaises(ocr_processor.OCRPreprocessingError) as cm:
            ocr_processor.preprocess_image_for_ocr("not_an_image")
        self.assertIn("输入必须是 PIL.Image.Image 对象。", str(cm.exception))

    # --- test_perform_ocr ---
    @patch('pytesseract.image_to_data')
    @patch('pytesseract.pytesseract.tesseract_cmd', new_callable=MagicMock) # 模拟 tesseract_cmd setter
    def test_perform_ocr_local_success(self, mock_tesseract_cmd_setter, mock_image_to_data):
        """测试本地OCR成功识别且置信度足够的情况。"""
        # 模拟 pytesseract.image_to_data 返回特定结果
        mock_image_to_data.return_value = {
            'text': ['Hello', 'World'],
            'conf': ['90', '85'],
            'left': [10, 60], 'top': [10, 10], 'width': [40, 40], 'height': [20, 20]
        }
        ocr_config = {'local_ocr_confidence_threshold': 70.0}
        
        text, confidence = ocr_processor.perform_ocr(self.dummy_image, ocr_config)
        self.assertEqual(text, "Hello World")
        self.assertAlmostEqual(confidence, (90 + 85) / 2) # 平均置信度
        mock_image_to_data.assert_called_once()
        # 确保如果 ocr_config 中未提供 tesseract_cmd，则 setter 未被调用
        mock_tesseract_cmd_setter.assert_not_called()

    @patch('pytesseract.image_to_data')
    @patch('pytesseract.pytesseract.tesseract_cmd', new_callable=MagicMock)
    def test_perform_ocr_local_low_confidence_fallback_to_cloud(self, mock_tesseract_cmd_setter, mock_image_to_data):
        """测试本地OCR置信度低，回退到云端OCR的情况。"""
        mock_image_to_data.return_value = {
            'text': ['low', 'conf'],
            'conf': ['30', '40'],
            'left': [10, 60], 'top': [10, 10], 'width': [40, 40], 'height': [20, 20]
        }
        ocr_config = {
            'local_ocr_confidence_threshold': 70.0,
            'cloud_ocr_enabled': True,
            'cloud_ocr_api_key': 'mock_key'
        }
        text, confidence = ocr_processor.perform_ocr(self.dummy_image, ocr_config)
        self.assertEqual(text, "这是模拟的云端 OCR 结果。 low conf") # 模拟云端结果会添加前缀
        self.assertAlmostEqual(confidence, 95.0) # 模拟云端置信度
        mock_image_to_data.assert_called_once()

    @patch('pytesseract.image_to_data')
    @patch('pytesseract.pytesseract.tesseract_cmd', new_callable=MagicMock)
    def test_perform_ocr_force_cloud(self, mock_tesseract_cmd_setter, mock_image_to_data):
        """测试强制使用云端OCR的情况。"""
        mock_image_to_data.return_value = {
            'text': ['high', 'conf'],
            'conf': ['90', '90'],
            'left': [10, 60], 'top': [10, 10], 'width': [40, 40], 'height': [20, 20]
        }
        ocr_config = {
            'local_ocr_confidence_threshold': 70.0,
            'force_cloud_ocr': True,
            'cloud_ocr_enabled': True,
            'cloud_ocr_api_key': 'mock_key'
        }
        text, confidence = ocr_processor.perform_ocr(self.dummy_image, ocr_config)
        self.assertEqual(text, "这是模拟的云端 OCR 结果。 high conf")
        self.assertAlmostEqual(confidence, 95.0)
        mock_image_to_data.assert_called_once()

    @patch('pytesseract.image_to_data', side_effect=pytesseract.TesseractNotFoundError)
    @patch('pytesseract.pytesseract.tesseract_cmd', new_callable=MagicMock)
    def test_perform_ocr_tesseract_not_found_with_cloud_fallback(self, mock_tesseract_cmd_setter, mock_image_to_data):
        """测试Tesseract未找到但有云端回退的情况。"""
        ocr_config = {
            'cloud_ocr_enabled': True,
            'cloud_ocr_api_key': 'mock_key'
        }
        text, confidence = ocr_processor.perform_ocr(self.dummy_image, ocr_config)
        self.assertEqual(text, "这是模拟的云端 OCR 结果。 ") # 模拟云端结果，本地OCR失败所以无文本
        self.assertAlmostEqual(confidence, 95.0)
        mock_image_to_data.assert_called_once()

    @patch('pytesseract.image_to_data', side_effect=pytesseract.TesseractNotFoundError)
    @patch('pytesseract.pytesseract.tesseract_cmd', new_callable=MagicMock)
    def test_perform_ocr_tesseract_not_found_no_cloud(self, mock_tesseract_cmd_setter, mock_image_to_data):
        """测试Tesseract未找到且未启用云端OCR的情况，应抛出错误。"""
        ocr_config = {
            'cloud_ocr_enabled': False
        }
        with self.assertRaises(ocr_processor.OCRPreprocessingError) as cm:
            ocr_processor.perform_ocr(self.dummy_image, ocr_config)
        self.assertIn("Tesseract 未找到且未启用云端 OCR", str(cm.exception))
        mock_image_to_data.assert_called_once()

    def test_perform_ocr_invalid_input(self):
        """测试perform_ocr的无效输入。"""
        with self.assertRaises(ocr_processor.OCRPreprocessingError) as cm:
            ocr_processor.perform_ocr("not_an_image", {})
        self.assertIn("输入必须是 PIL.Image.Image 对象。", str(cm.exception))
        
        with self.assertRaises(ocr_processor.OCRPreprocessingError) as cm:
            ocr_processor.perform_ocr(self.dummy_image, "not_a_dict")
        self.assertIn("输入 ocr_config 必须是字典。", str(cm.exception))

    # --- test_clean_and_structure_chat_text ---
    @patch('pytesseract.image_to_data')
    def test_clean_and_structure_chat_text_basic(self, mock_image_to_data):
        """测试基本的聊天文本清洗和结构化，包括发言人识别。"""
        # 模拟 image_to_data 返回两个发言人的文本和位置
        # '对方'在左侧，'我'在右侧
        mock_image_to_data.return_value = {
            'level': [1, 2, 3, 4, 5, 6],
            'page_num': [1, 1, 1, 1, 1, 1],
            'block_num': [1, 1, 1, 2, 2, 3], # 模拟块，以确保分组
            'par_num': [1, 1, 1, 1, 1, 1],
            'line_num': [1, 1, 1, 1, 1, 1],
            'word_num': [1, 2, 3, 1, 2, 1],
            # left 坐标：50, 80, 110 (左侧) -> 对方; 200, 230 (右侧) -> 我; 60 (左侧) -> 对方
            'left': [50, 80, 110, 200, 230, 60], 
            'top': [10, 10, 10, 30, 30, 50],
            'width': [30, 30, 30, 30, 30, 30],
            'height': [20, 20, 20, 20, 20, 20],
            'conf': ['90', '90', '90', '90', '90', '90'],
            'text': ['你好', '吗', '朋友', '我', '很好', '再见']
        }
        
        # 创建一个宽度大于100的虚拟图像，以便进行发言人检测
        image_for_test = create_dummy_image(width=400, height=100)
        
        # raw_text 在此函数中不直接用于内容解析，而是用于日志或回退情况
        raw_text_placeholder = "你好吗朋友 我很好 再见" 
        
        structured_messages = ocr_processor.clean_and_structure_chat_text(raw_text_placeholder, image_for_test)
        
        expected_messages = [
            f"{ocr_processor.SPEAKER_OTHER}: 你好 吗 朋友",
            f"{ocr_processor.SPEAKER_ME}: 很好", # <--- 修正点：这里不再包含冗余的“我”
            f"{ocr_processor.SPEAKER_OTHER}: 再见"
        ]
        
        self.assertEqual(structured_messages, expected_messages)
        mock_image_to_data.assert_called_once()

    @patch('pytesseract.image_to_data')
    def test_clean_and_structure_chat_text_with_noise(self, mock_image_to_data):
        """测试带有噪声（时间戳、占位符）的聊天文本清洗。"""
        mock_image_to_data.return_value = {
            'level': [1, 2, 3, 4, 5, 6, 7, 8],
            'page_num': [1, 1, 1, 1, 1, 1, 1, 1], # 修正 block_num 为 page_num
            'block_num': [1, 1, 1, 1, 2, 2, 2, 3], # 修正 block_num
            'par_num': [1, 1, 1, 1, 1, 1, 1, 1],
            'line_num': [1, 1, 1, 1, 1, 1, 1, 1],
            'word_num': [1, 2, 3, 4, 1, 2, 3, 1],
            'left': [50, 80, 120, 150, 200, 230, 260, 50],
            'top': [10, 10, 10, 10, 30, 30, 30, 50],
            'width': [30, 30, 30, 30, 30, 30, 30, 30],
            'height': [20, 20, 20, 20, 20, 20, 20, 20],
            'conf': ['90', '90', '90', '90', '90', '90', '90', '90'],
            'text': ['上午', '10:30', '你好', '朋友', '我', '很好', '[图片]', '再见']
        }
        
        image_for_test = create_dummy_image(width=400, height=100)
        raw_text_placeholder = "上午 10:30 你好朋友 我很好 [图片] 再见"
        
        structured_messages = ocr_processor.clean_and_structure_chat_text(raw_text_placeholder, image_for_test)
        
        expected_messages = [
            f"{ocr_processor.SPEAKER_OTHER}: 你好 朋友", # "上午 10:30" 被移除
            f"{ocr_processor.SPEAKER_ME}: 很好",   # "[图片]" 被移除，并且“我”也被移除
            f"{ocr_processor.SPEAKER_OTHER}: 再见"
        ]
        
        self.assertEqual(structured_messages, expected_messages)
        mock_image_to_data.assert_called_once()

    @patch('pytesseract.image_to_data', side_effect=pytesseract.TesseractNotFoundError)
    def test_clean_and_structure_chat_text_tesseract_not_found_fallback(self, mock_image_to_data):
        """测试Tesseract未找到时，clean_and_structure_chat_text的回退行为。"""
        raw_text = "Hello World [图片] 2023年10月26日"
        image_for_test = create_dummy_image()
        
        structured_messages = ocr_processor.clean_and_structure_chat_text(raw_text, image_for_test)
        
        # 期望一个带有“未知发言人”和已清洗文本的单条消息
        self.assertEqual(structured_messages, ["未知发言人: Hello World"])
        mock_image_to_data.assert_called_once()

    def test_clean_and_structure_chat_text_invalid_input(self):
        """测试clean_and_structure_chat_text的无效输入。"""
        with self.assertRaises(ocr_processor.OCRPreprocessingError) as cm:
            ocr_processor.clean_and_structure_chat_text(123, self.dummy_image)
        self.assertIn("输入 raw_text 必须是字符串。", str(cm.exception))

        with self.assertRaises(ocr_processor.OCRPreprocessingError) as cm:
            ocr_processor.clean_and_structure_chat_text("some text", "not_an_image")
        self.assertIn("输入 image 必须是 PIL.Image.Image 对象。", str(cm.exception))

    # --- test_clean_chat_message_helper ---
    def test_clean_chat_message_helper(self):
        """测试辅助函数 _clean_chat_message 的清洗逻辑。"""
        # 调用 ocr_processor._clean_chat_message
        self.assertEqual(ocr_processor._clean_chat_message("上午 10:30 你好"), "你好")
        self.assertEqual(ocr_processor._clean_chat_message("2023年10月26日 10:30 消息内容"), "消息内容")
        self.assertEqual(ocr_processor._clean_chat_message("消息内容 [图片]"), "消息内容")
        self.assertEqual(ocr_processor._clean_chat_message("消息内容 [语音] [表情]"), "消息内容")
        self.assertEqual(ocr_processor._clean_chat_message("  多 余  空 格   "), "多 余 空 格")
        self.assertEqual(ocr_processor._clean_chat_message(""), "")
        self.assertEqual(ocr_processor._clean_chat_message("[图片] [语音]"), "") # 清洗后应为空字符串
        self.assertEqual(ocr_processor._clean_chat_message("10月26日 消息"), "消息")
        self.assertEqual(ocr_processor._clean_chat_message("2024年1月1日"), "")
        self.assertEqual(ocr_processor._clean_chat_message("晚上 8:00 晚安"), "晚安")
        self.assertEqual(ocr_processor._clean_chat_message("Hello world."), "Hello world.") # 无需清洗的文本

        # --- 新增测试：测试冗余发言人名称的移除 ---
        self.assertEqual(ocr_processor._clean_chat_message("我 很好", ocr_processor.SPEAKER_ME), "很好")
        self.assertEqual(ocr_processor._clean_chat_message("对方：你好", ocr_processor.SPEAKER_OTHER), "你好") # <--- 修正点：这个测试应该通过了
        self.assertEqual(ocr_processor._clean_chat_message("我: 我很好", ocr_processor.SPEAKER_ME), "我很好") # 冒号后不移除
        self.assertEqual(ocr_processor._clean_chat_message("你好，我很好", ocr_processor.SPEAKER_ME), "你好，我很好") # 不在开头不移除
        self.assertEqual(ocr_processor._clean_chat_message("张三：你好", "张三"), "你好") # 模拟具体人名
        self.assertEqual(ocr_processor._clean_chat_message("我就是我", ocr_processor.SPEAKER_ME), "就是我") # 确保只移除一次
        # ------------------------------------------

# 如果直接运行此文件，则执行所有测试
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)