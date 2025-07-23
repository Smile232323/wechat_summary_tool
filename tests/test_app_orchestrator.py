import unittest
from unittest.mock import patch, MagicMock
import os
from PIL import Image
import pyautogui # 确保这里也导入了 pyautogui，以便在测试中直接使用其类型
import logging
import sys

# --- 关键修正：确保能正确导入 src/gui_automation/wechat_gui_automator ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui_automation import wechat_gui_automator
# ------------------------------------------------------------------

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestWechatGUIAutomator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """在所有测试运行前执行一次，用于设置全局配置。"""
        logging.getLogger('src.gui_automation.wechat_gui_automator').setLevel(logging.CRITICAL)
        logging.getLogger('pyautogui').setLevel(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        """在所有测试运行后执行一次，用于清理全局配置。"""
        logging.getLogger('src.gui_automation.wechat_gui_automator').setLevel(logging.INFO)
        logging.getLogger('pyautogui').setLevel(logging.INFO)

    def setUp(self):
        # 模拟一个虚拟的微信窗口对象
        self.mock_wechat_window = MagicMock(spec=pyautogui.Window) # 使用 spec 确保模拟对象有真实对象的属性和方法
        self.mock_wechat_window.title = "微信"
        self.mock_wechat_window.left = 100
        self.mock_wechat_window.top = 50
        self.mock_wechat_window.width = 800
        self.mock_wechat_window.height = 600
        # mock_wechat_window.activate 方法会被自动创建，因为使用了 spec

    # --- test_find_wechat_window ---
    # Patch pyautogui.getWindowsWithTitle 在 wechat_gui_automator 模块中被引用的位置
    @patch('src.gui_automation.wechat_gui_automator.pyautogui.getWindowsWithTitle')
    def test_find_wechat_window_found(self, mock_get_windows):
        """测试找到微信窗口的情况。"""
        mock_get_windows.return_value = [self.mock_wechat_window]
        window = wechat_gui_automator.find_wechat_window()
        self.assertIsNotNone(window)
        self.assertEqual(window.title, "微信")
        mock_get_windows.assert_called_once_with("微信")
        # 直接断言 mock 对象的 activate 方法
        self.mock_wechat_window.activate.assert_called_once() 

    @patch('src.gui_automation.wechat_gui_automator.pyautogui.getWindowsWithTitle')
    def test_find_wechat_window_not_found(self, mock_get_windows):
        """测试未找到微信窗口的情况。"""
        mock_get_windows.return_value = []
        window = wechat_gui_automator.find_wechat_window()
        self.assertIsNone(window)
        mock_get_windows.assert_called_once_with("微信")
        self.mock_wechat_window.activate.assert_not_called() # 确保未找到时不调用激活

    # --- test_capture_chat_area ---
    # Patch pyautogui.screenshot 在 wechat_gui_automator 模块中被引用的位置
    @patch('src.gui_automation.wechat_gui_automator.pyautogui.screenshot')
    def test_capture_chat_area_success(self, mock_screenshot):
        """测试成功截取聊天区域。"""
        mock_screenshot.return_value = Image.new('RGB', (100, 100)) # 模拟返回一个PIL Image
        test_region = (10, 10, 100, 100)
        screenshot = wechat_gui_automator.capture_chat_area(self.mock_wechat_window, test_region)
        self.assertIsInstance(screenshot, Image.Image)
        mock_screenshot.assert_called_once_with(region=test_region)

    @patch('src.gui_automation.wechat_gui_automator.pyautogui.screenshot', side_effect=Exception("Screenshot error"))
    def test_capture_chat_area_failure(self, mock_screenshot):
        """测试截取聊天区域失败。"""
        test_region = (10, 10, 100, 100)
        # 预期不抛出异常，而是返回 None
        screenshot = wechat_gui_automator.capture_chat_area(self.mock_wechat_window, test_region)
        self.assertIsNone(screenshot)
        mock_screenshot.assert_called_once_with(region=test_region)

    def test_capture_chat_area_invalid_input(self):
        """测试无效输入的 capture_chat_area。"""
        # 预期抛出自定义的 GUIAutomationError
        with self.assertRaises(wechat_gui_automator.GUIAutomationError) as cm:
            wechat_gui_automator.capture_chat_area(self.mock_wechat_window, "invalid_region")
        self.assertIn("region 必须是包含4个整数的元组", str(cm.exception))

        with self.assertRaises(wechat_gui_automator.GUIAutomationError) as cm:
            wechat_gui_automator.capture_chat_area("not_a_window", (0,0,10,10))
        self.assertIn("window_handle 必须是 pyautogui.Window 对象", str(cm.exception))

    # --- test_scroll_chat_window ---
    # Patch pyautogui.scroll 在 wechat_gui_automator 模块中被引用的位置
    @patch('src.gui_automation.wechat_gui_automator.pyautogui.scroll')
    def test_scroll_chat_window_success(self, mock_scroll):
        """测试成功滚动聊天窗口。"""
        wechat_gui_automator.scroll_chat_window(self.mock_wechat_window, scroll_amount=-500)
        # 计算预期的 x, y 坐标
        expected_x = self.mock_wechat_window.left + self.mock_wechat_window.width // 2
        expected_y = self.mock_wechat_window.top + self.mock_wechat_window.height // 2
        mock_scroll.assert_called_once_with(-500, x=expected_x, y=expected_y) # 确保参数匹配

    @patch('src.gui_automation.wechat_gui_automator.pyautogui.scroll', side_effect=Exception("Scroll error"))
    def test_scroll_chat_window_failure(self, mock_scroll):
        """测试滚动聊天窗口失败。"""
        # 预期不抛出异常，而是记录错误
        try:
            wechat_gui_automator.scroll_chat_window(self.mock_wechat_window, scroll_amount=-500)
        except Exception as e:
            self.fail(f"scroll_chat_window 意外抛出异常: {e}")
        mock_scroll.assert_called_once() # 确保被调用了

    def test_scroll_chat_window_invalid_input(self):
        """测试无效输入的 scroll_chat_window。"""
        with self.assertRaises(wechat_gui_automator.GUIAutomationError) as cm:
            wechat_gui_automator.scroll_chat_window("not_a_window", -500)
        self.assertIn("window_handle 必须是 pyautogui.Window 对象", str(cm.exception))

        with self.assertRaises(wechat_gui_automator.GUIAutomationError) as cm:
            wechat_gui_automator.scroll_chat_window(self.mock_wechat_window, "not_an_int")
        self.assertIn("scroll_amount 必须是整数", str(cm.exception))

    # --- test_detect_scroll_end ---
    def test_detect_scroll_end_identical_images(self):
        """测试相同图像，应返回 True (已到底部)。"""
        image1 = Image.new('RGB', (100, 100), color='red')
        image2 = Image.new('RGB', (100, 100), color='red')
        self.assertTrue(wechat_gui_automator.detect_scroll_end(image1, image2))

    def test_detect_scroll_end_different_images(self):
        """测试不同图像，应返回 False (未到底部)。"""
        image1 = Image.new('RGB', (100, 100), color='red')
        image2 = Image.new('RGB', (100, 100), color='blue')
        self.assertFalse(wechat_gui_automator.detect_scroll_end(image1, image2))

    def test_detect_scroll_end_invalid_input(self):
        """测试 detect_scroll_end 的无效输入。"""
        image1 = Image.new('RGB', (100, 100))
        with self.assertRaises(wechat_gui_automator.GUIAutomationError) as cm:
            wechat_gui_automator.detect_scroll_end("not_an_image", image1)
        self.assertIn("必须是 PIL.Image.Image 对象", str(cm.exception))

        with self.assertRaises(wechat_gui_automator.GUIAutomationError) as cm:
            wechat_gui_automator.detect_scroll_end(image1, "not_an_image")
        self.assertIn("必须是 PIL.Image.Image 对象", str(cm.exception))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)