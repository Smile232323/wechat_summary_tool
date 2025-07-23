# src/gui_automation/wechat_gui_automator.py (终极识别版 - 修正 - 修复打包后截图目录问题)
import pyautogui
import logging
import time
import os
import sys # 新增导入 sys 模块
from PIL import Image
from typing import Optional, Tuple, List
import cv2
import numpy as np

# --- 导入新的依赖 ---
try:
    import win32gui
    import win32con # 导入win32con用于窗口状态常量
except ImportError:
    win32gui = None
    win32con = None
    logging.warning("pywin32 未安装，将无法使用基于窗口类名的查找方法。请运行 'pip install pywin32'。")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 微信主窗口的类名，这是比标题更可靠的标识符
WECHAT_CLASS_NAME = "WeChatMainWndForPC"

class GUIAutomationError(Exception):
    pass

def find_wechat_window() -> Optional[pyautogui.Window]:
    """
    【终极修正】
    通过“类名”和“标题”双重保险来查找微信窗口，并确保其激活。
    返回一个 pyautogui.Window 对象。
    """
    wechat_pyautogui_window = None

    # --- 策略一：通过窗口类名（最可靠）---
    if win32gui and win32con:
        try:
            logger.info(f"尝试通过类名 '{WECHAT_CLASS_NAME}' 查找微信窗口...")
            hwnd = win32gui.FindWindow(WECHAT_CLASS_NAME, None)
            if hwnd != 0:
                logger.info(f"通过类名找到窗口句柄: {hwnd}")
                
                # 确保窗口不是最小化状态，并将其恢复
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                
                # 将窗口置于前台并激活
                win32gui.SetForegroundWindow(hwnd)
                time.sleep(0.5) # 给予系统时间来激活窗口

                # 尝试获取当前活动窗口，并验证是否是微信窗口
                # 此时，pyautogui.getActiveWindow() 应该能正确获取到我们刚刚激活的窗口
                active_window = pyautogui.getActiveWindow()
                if active_window and ('微信' in active_window.title or 'WeChat' in active_window.title or getattr(active_window, '_hWnd', None) == hwnd):
                    # 检查_hWnd是更底层的，如果pyautogui能获取到，它会有一个_hWnd属性
                    logger.info(f"成功通过类名定位并激活微信窗口：'{active_window.title}'！")
                    return active_window
                else:
                    logger.warning(f"通过类名激活后，pyautogui未能识别为预期微信窗口。当前活动窗口: '{active_window.title if active_window else 'None'}'。将回退到标题查找法。")
            else:
                logger.info(f"未通过类名 '{WECHAT_CLASS_NAME}' 找到窗口。")
        except Exception as e:
            logger.warning(f"通过类名查找或激活时发生错误: {e}。将回退到标题查找法。", exc_info=True)
    else:
        logger.warning("pywin32 未安装或导入失败，无法使用类名查找。")

    # --- 策略二：通过窗口标题包含“微信”（作为备用方案）---
    logger.info("类名查找失败或不可用，回退到通过标题查找...")
    possible_titles_parts = ["微信", "WeChat", "文件传输助手", "Weixin", "WeChat PC"]
    for title_part in possible_titles_parts:
        windows = pyautogui.getWindowsWithTitle(title_part)
        for window in windows:
            if window.title and window.title != '': # 确保标题不为空
                try:
                    if window.isMinimized:
                        window.restore()
                    window.activate()
                    logger.info(f"通过标题 '{window.title}' 找到并激活微信窗口。")
                    return window
                except Exception as e:
                    logger.warning(f"激活窗口 '{window.title}' 时发生错误: {e}")
    
    logger.error("所有方法均失败，未找到微信窗口。")
    return None

def capture_chat_area(window_handle: pyautogui.Window, region: Optional[Tuple[int, int, int, int]]) -> Optional[Image.Image]:
    """
    强制全屏截图，忽略传入的 region 参数。
    """
    try:
        screen_width, screen_height = pyautogui.size()
        full_screen_region = (0, 0, screen_width, screen_height)
        screenshot = pyautogui.screenshot(region=full_screen_region)
        logger.info(f"已截取全屏区域。")
        return screenshot
    except Exception as e:
        logger.error(f"截取全屏时发生错误: {e}", exc_info=True)
        return None

def scroll_chat_window(window_handle: pyautogui.Window, scroll_amount: int = -600) -> None:
    """
    滚动当前激活的微信窗口。
    为了确保滚动有效，将鼠标移动到窗口中心再执行滚动。
    """
    try:
        # 移动鼠标到窗口中心，确保滚动操作作用于此窗口
        pyautogui.moveTo(window_handle.left + window_handle.width // 2, window_handle.top + window_handle.height // 2)
        pyautogui.scroll(scroll_amount)
        logger.info(f"已滚动 {scroll_amount} 像素。")
    except Exception as e:
        logger.error(f"滚动聊天窗口时发生错误: {e}", exc_info=True)

def detect_scroll_end(current_image: Image.Image, previous_image: Image.Image, threshold: float = 0.80) -> bool: # 阈值改为 0.80
    """
    通过结构相似性指数 (SSIM) 比较两张图片，判断是否已滚动到末尾。
    当SSIM分数高于给定的阈值时，认为已达到末尾。
    """
    if current_image.size != previous_image.size: 
        return False
    try:
        # 确保图像是灰度图，且数据类型正确
        img1_np = np.array(current_image.convert('L'))
        img2_np = np.array(previous_image.convert('L'))
        
        # 使用SSIM进行比较
        (score, diff) = cv2.compareSSIM(img1_np, img2_np, full=True)
        
        # 打印SSIM分数，帮助调试
        logger.info(f"SSIM score between current and previous image: {score:.4f} (Threshold: {threshold:.2f})")
        
        # 相似度阈值，根据您的要求设置为0.80
        if score > threshold: 
            logger.info(f"检测到图像内容高度相似 (SSIM: {score:.4f} > {threshold:.2f})，滚动结束。")
            return True
        return False
    except Exception as e:
        logger.error(f"检测滚动结束时发生错误: {e}", exc_info=True)
        return False

def capture_chat_history_dynamically(
    region: Optional[Tuple[int, int, int, int]], # 此参数现在是多余的，但保留以兼容旧调用
    screenshot_dir: str, # 传入的是 "temp_screenshots"
    max_scrolls: int = 20,
    scroll_amount: int = -600,
    delay_between_scrolls: float = 0.5,
    similarity_threshold: float = 0.80
) -> List[str]:
    """
    动态捕获聊天记录，通过滚动和截图实现。
    """
    wechat_win = find_wechat_window()
    if not wechat_win:
        raise GUIAutomationError("未找到微信窗口，无法开始截图。")

    # --- 关键修改：确定截图目录的绝对路径 ---
    if getattr(sys, 'frozen', False): # 如果程序是PyInstaller打包的
        # base_path 是 .exe 文件所在的目录
        base_path = os.path.dirname(sys.executable)
    else:
        # 如果是直接运行Python脚本
        # base_path 是 main_app.py 所在的目录 (项目根目录)
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # 或者更简单，如果确保是从项目根目录运行的：base_path = os.getcwd()

    # 拼接出完整的截图目录绝对路径
    absolute_screenshot_dir = os.path.join(base_path, screenshot_dir) # screenshot_dir 仍然是 "temp_screenshots"

    if not os.path.exists(absolute_screenshot_dir):
        try:
            os.makedirs(absolute_screenshot_dir)
            logger.info(f"Created screenshot directory: {absolute_screenshot_dir}")
        except OSError as e:
            logger.error(f"Failed to create screenshot directory {absolute_screenshot_dir}: {e}", exc_info=True)
            # 抛出更具体的错误，方便用户诊断
            raise GUIAutomationError(f"无法创建截图目录: {absolute_screenshot_dir}. 请检查文件系统权限或路径是否有效。") from e
    # --- 关键修改结束 ---

    image_paths = []
    previous_image = None
    for i in range(max_scrolls):
        logger.info(f"--- 智能滚动第 {i + 1}/{max_scrolls} 轮 ---")
        
        current_image = capture_chat_area(wechat_win, None)
        if not current_image:
            logger.warning("本次截图失败，终止滚动。")
            break
        
        if previous_image and detect_scroll_end(current_image, previous_image, similarity_threshold):
            logger.info("检测到图片内容高度相似，提前停止截图，不保存当前重复图片。")
            break
        
        # 将图片保存到绝对路径
        file_path = os.path.join(absolute_screenshot_dir, f"capture_{i + 1}.png")
        current_image.save(file_path)
        image_paths.append(file_path)
        logger.info(f"截图已保存至 {file_path}")

        previous_image = current_image
        
        if i < max_scrolls - 1:
            scroll_chat_window(wechat_win, scroll_amount)
            time.sleep(delay_between_scrolls)
    else:
        logger.warning(f"已达到最大滚动次数 ({max_scrolls})，自动停止。")
    return image_paths