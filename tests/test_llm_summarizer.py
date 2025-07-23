import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import logging

# 确保能正确导入 src/llm_integration/llm_summarizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从被测试模块导入其使用的异常类，确保一致性
from src.llm_integration import llm_summarizer
from src.llm_integration.llm_summarizer import LLMSummarizationError, GoogleAPIError, BlockedPromptException

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestLLMSummarizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试运行前执行一次，用于设置全局配置。"""
        # 在测试期间将模块日志级别调高，避免控制台输出过多无关信息
        logging.getLogger('src.llm_integration.llm_summarizer').setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        """在所有测试运行后执行一次，用于清理全局配置。"""
        logging.getLogger('src.llm_integration.llm_summarizer').setLevel(logging.INFO)

    def setUp(self):
        self.mock_api_key = "test_api_key"
        self.sample_raw_text = "This is some raw text with errors. [Image] 10:30 AM."
        self.sample_chat_messages = [
            "我: 你好，今天开会吗？",
            "对方: 是的，下午三点。地点在会议室A。",
            "我: 好的，我准备一下。",
            "对方: 收到[表情]"
        ]
        # 使长消息更长，以确保分段逻辑被触发
        self.long_chat_messages = [f"Message {i}: This is a long message to ensure segmentation logic is triggered." for i in range(100)]

    # --- test_llm_assisted_text_cleaning ---
    @patch('src.llm_integration.llm_summarizer.genai.GenerativeModel')
    @patch('src.llm_integration.llm_summarizer.genai.configure')
    def test_llm_assisted_text_cleaning_success(self, mock_configure, MockGenerativeModel):
        """测试LLM辅助文本清洗成功。"""
        mock_model_instance = MockGenerativeModel.return_value
        mock_response = MagicMock()
        mock_response.text = "This is some cleaned text. No errors."
        mock_model_instance.generate_content.return_value = mock_response

        cleaned_text = llm_summarizer.llm_assisted_text_cleaning(self.sample_raw_text, self.mock_api_key)
        
        self.assertEqual(cleaned_text, "This is some cleaned text. No errors.")
        mock_configure.assert_called_once_with(api_key=self.mock_api_key)
        MockGenerativeModel.assert_called_once_with(llm_summarizer.DEFAULT_LLM_MODEL)
        mock_model_instance.generate_content.assert_called_once()

    @patch('src.llm_integration.llm_summarizer.genai.GenerativeModel')
    @patch('src.llm_integration.llm_summarizer.genai.configure')
    def test_llm_assisted_text_cleaning_api_failure(self, mock_configure, MockGenerativeModel):
        """测试LLM辅助文本清洗API调用失败。"""
        MockGenerativeModel.return_value.generate_content.side_effect = GoogleAPIError("API Error")
        
        with self.assertRaises(LLMSummarizationError) as cm:
            llm_summarizer.llm_assisted_text_cleaning(self.sample_raw_text, self.mock_api_key)
        self.assertIn("API Error", str(cm.exception))

    # --- test_summarize_chat_history ---
    @patch('src.llm_integration.llm_summarizer.genai.GenerativeModel')
    @patch('src.llm_integration.llm_summarizer.genai.configure')
    def test_summarize_chat_history_short_success(self, mock_configure, MockGenerativeModel):
        """测试短聊天记录总结成功。"""
        mock_model_instance = MockGenerativeModel.return_value
        mock_model_instance.generate_content.return_value = MagicMock(text="Summary of short chat.")

        summary = llm_summarizer.summarize_chat_history(self.sample_chat_messages, self.mock_api_key)
        
        self.assertEqual(summary, "Summary of short chat.")
        mock_configure.assert_called_once_with(api_key=self.mock_api_key)
        mock_model_instance.generate_content.assert_called_once()
        prompt_arg = mock_model_instance.generate_content.call_args[0][0]
        self.assertIn("请总结以下微信聊天记录", prompt_arg)
        self.assertIn("\n".join(self.sample_chat_messages), prompt_arg)

    @patch('src.llm_integration.llm_summarizer.genai.GenerativeModel')
    @patch('src.llm_integration.llm_summarizer.genai.configure')
    def test_summarize_chat_history_long_success(self, mock_configure, MockGenerativeModel):
        """测试长聊天记录分段总结成功（使用可调用side_effect）。"""
        mock_model_instance = MockGenerativeModel.return_value
        
        def mock_generate_content_logic(prompt):
            if "请整合以下分段总结" in prompt:
                return MagicMock(text="Final Combined Summary.")
            else:
                return MagicMock(text="Segment Summary.")

        mock_model_instance.generate_content.side_effect = mock_generate_content_logic

        summary = llm_summarizer.summarize_chat_history(self.long_chat_messages, self.mock_api_key, max_tokens=100)

        self.assertEqual(summary, "Final Combined Summary.")
        mock_configure.assert_called_once_with(api_key=self.mock_api_key)
        self.assertGreater(mock_model_instance.generate_content.call_count, 1)
        last_call_args = mock_model_instance.generate_content.call_args[0]
        self.assertIn("请整合以下分段总结", last_call_args[0])
        self.assertIn("Segment Summary.", last_call_args[0])

    @patch('src.llm_integration.llm_summarizer.genai.GenerativeModel')
    @patch('src.llm_integration.llm_summarizer.genai.configure')
    def test_summarize_chat_history_api_failure(self, mock_configure, MockGenerativeModel):
        """测试LLM总结API调用失败。"""
        MockGenerativeModel.return_value.generate_content.side_effect = GoogleAPIError("API Error")
        
        with self.assertRaises(LLMSummarizationError) as cm:
            llm_summarizer.summarize_chat_history(self.sample_chat_messages, self.mock_api_key)
        self.assertIn("API Error", str(cm.exception))

    def test_summarize_chat_history_invalid_input(self):
        """测试 summarize_chat_history 的无效输入。"""
        with self.assertRaises(LLMSummarizationError):
            llm_summarizer.summarize_chat_history("not_a_list", self.mock_api_key)
        with self.assertRaises(LLMSummarizationError):
            llm_summarizer.summarize_chat_history([1, 2, 3], self.mock_api_key)
        with self.assertRaises(LLMSummarizationError):
            llm_summarizer.summarize_chat_history(self.sample_chat_messages, None)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)