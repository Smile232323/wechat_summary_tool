# main_app.py (最终完全体 - 合并版 - 修复bug - 强制全屏截图 - 窗口自动隐藏)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Dict, Callable, List, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# --- 导入所有需要的真实后端模块 ---
# 注意：find_wechat_window 将不再直接使用，但 wechat_gui_automator 仍然需要导入
from src.gui_automation.wechat_gui_automator import capture_chat_history_dynamically, GUIAutomationError
from src.ocr_processing.ocr_processor import clean_and_structure_chat_text, OCRPreprocessingError
from PIL import Image, ImageFilter
from difflib import SequenceMatcher
# --- 导入两个LLM库 ---
import dashscope
import google.generativeai as genai

# --- 日志配置 ---
LOG_FILE = "wechat_summary_tool.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- 【适配器】LLMManager 现在是调度中心，负责调用正确的LLM ---
class LLMManager:
    def __init__(self, log_callback: Callable):
        self.log_callback = log_callback

    def summarize(self, api_service: str, chat_history: List[str], api_key: str) -> str:
        self.log_callback("INFO", f"使用 {api_service} 进行总结...")
        
        if api_service == '阿里云百炼':
            return self._summarize_with_dashscope(chat_history, api_key)
        elif api_service == 'Google Gemini':
            return self._summarize_with_gemini(chat_history, api_key)
        else:
            raise ValueError(f"不支持的API服务: {api_service}")

    def _summarize_with_dashscope(self, chat_history: List[str], api_key: str) -> str:
        if not api_key: raise ValueError("阿里云百炼 API Key 未提供。")
        dashscope.api_key = api_key
        # 将聊天记录列表合并成一个字符串作为prompt
        prompt = f"请总结以下微信聊天记录的关键信息、决策点和待办事项。请以结构化的方式呈现。\n\n聊天记录：\n{''.join(chat_history)}"
        try:
            response = dashscope.Generation.call(model="qwen-plus", prompt=prompt, result_format='text')
            if response.status_code == 200:
                return response.output['text'].strip()
            else:
                raise ConnectionError(f"阿里云百炼API错误: {response.message} (Code: {response.status_code})")
        except Exception as e:
            raise ConnectionError(f"调用阿里云百炼API时发生错误: {e}")

    def _summarize_with_gemini(self, chat_history: List[str], api_key: str) -> str:
        if not api_key: raise ValueError("Google Gemini API Key 未提供。")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        # 将聊天记录列表合并成一个字符串作为prompt
        prompt = f"请总结以下微信聊天记录的关键信息、决策点和待办事项。请以结构化的方式呈现。\n\n聊天记录：\n{''.join(chat_history)}"
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise ConnectionError(f"调用Google Gemini API时发生错误: {e}")

# --- OCR适配器 (从集成版引入，用于封装OCR处理逻辑) ---
class RealOCRAI:
    """适配器: 连接GUI和真实的OCR处理模块"""
    def __init__(self, log_callback: Callable):
        self.log_callback = log_callback

    def process_images(self, image_paths: List[str]) -> List[str]:
        self.log_callback("INFO", "调用真实OCR模块...")
        all_chat_lines = []
        for i, path in enumerate(image_paths):
            self.log_callback("INFO", f"  - 正在处理图片 {i+1}/{len(image_paths)}: {os.path.basename(path)}")
            try:
                with Image.open(path) as img:
                    # clean_and_structure_chat_text 期望 raw_text 或 image
                    structured_lines = clean_and_structure_chat_text(raw_text="", image=img)
                    all_chat_lines.extend(structured_lines)
            except Exception as e:
                self.log_callback("ERROR", f"处理图片 {path} 时失败: {e}")
                # 如果单个图片处理失败，不中断整个流程，但记录错误
        return all_chat_lines

# --- 主应用程序 ---
class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("聊天记录自动化总结工具 (全屏截图版)")
        self.geometry("850x800")
        self.resizable(True, True)

        self.config_file = "config.json"
        self.config: Dict[str, Any] = {}
        self.running_process_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        self.llm_manager = LLMManager(self.log_message)
        self.ocr_ai = RealOCRAI(self.log_message) # 实例化OCR适配器

        self.create_widgets()
        self.load_configuration()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 配置主窗口的网格布局
        self.grid_rowconfigure(1, weight=1) # 使日志/总结区域可扩展
        self.grid_columnconfigure(0, weight=1)

        # 配置框架
        config_frame = ttk.LabelFrame(self, text="配置", padding="10")
        config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        config_frame.grid_columnconfigure(1, weight=1) # 使输入框可扩展

        # --- API服务选择 ---
        ttk.Label(config_frame, text="选择AI服务:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.api_service_var = tk.StringVar()
        self.api_service_combo = ttk.Combobox(config_frame, textvariable=self.api_service_var, values=['阿里云百炼', 'Google Gemini'], state="readonly")
        self.api_service_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # --- 阿里云百炼API Key ---
        ttk.Label(config_frame, text="百炼API Key:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.bailian_api_key_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.bailian_api_key_var, show="*").grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # --- Google Gemini API Key ---
        ttk.Label(config_frame, text="Gemini API Key:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.gemini_api_key_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.gemini_api_key_var, show="*").grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # --- 新增: 图片压缩复选框 ---
        self.compress_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(config_frame, text="压缩图片(用于云端OCR)", variable=self.compress_var).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        
        # --- 最大滚动次数 (调整行号) ---
        ttk.Label(config_frame, text="最大滚动次数:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.scrolls_var = tk.StringVar(value="15")
        ttk.Entry(config_frame, textvariable=self.scrolls_var, width=5).grid(row=4, column=1, padx=(0,0), pady=5, sticky="w") # 调整padx

        # --- 按钮框架 (调整行号) ---
        button_frame = ttk.Frame(self, padding="5")
        button_frame.grid(row=5, column=0, pady=5) # 调整行号
        self.start_button = ttk.Button(button_frame, text="开始总结", command=self.start_summary_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="停止", command=self.stop_summary_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.save_config_button = ttk.Button(button_frame, text="保存配置", command=self.save_configuration)
        self.save_config_button.pack(side=tk.LEFT, padx=5)

        # --- 日志与总结结果区域 ---
        output_frame = ttk.LabelFrame(self, text="日志与总结结果", padding="10")
        output_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(output_frame, wrap="word", state="disabled", height=15)
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def log_message(self, level: str, message: str, exc_info: bool = False):
        """向日志文件和GUI日志区域输出消息"""
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message, exc_info=exc_info)
        self.after(0, self._update_gui_log, level, message)

    def _update_gui_log(self, level: str, message: str):
        """更新GUI日志区域"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END) # 自动滚动到最新消息

    def load_configuration(self):
        """加载保存的配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {} # 如果文件不存在，则为空配置

            # 加载API Key
            self.bailian_api_key_var.set(self.config.get("bailian_api_key", os.getenv("DASHSCOPE_API_KEY", "")))
            self.gemini_api_key_var.set(self.config.get("gemini_api_key", os.getenv("GOOGLE_API_KEY", "")))
            self.api_service_var.set(self.config.get("selected_api", "阿里云百炼"))
            
            # 加载其他配置
            self.compress_var.set(self.config.get("compress_images", False))
            self.scrolls_var.set(self.config.get("max_scrolls", 15))
            
            self.log_message("INFO", "配置加载成功。")
        except Exception as e:
            self.log_message("ERROR", f"加载配置失败: {e}", exc_info=True)

    def save_configuration(self):
        """保存当前配置"""
        try:
            self.config["selected_api"] = self.api_service_var.get()
            self.config["bailian_api_key"] = self.bailian_api_key_var.get()
            self.config["gemini_api_key"] = self.gemini_api_key_var.get()
            self.config["compress_images"] = self.compress_var.get()
            self.config["max_scrolls"] = self.scrolls_var.get()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            self.log_message("INFO", "配置已保存。")
        except Exception as e:
            self.log_message("ERROR", f"保存配置失败: {e}", exc_info=True)
            
    def _compress_image(self, image_path: str, quality: int = 75) -> str:
        """
        压缩图片功能。
        将PNG图片以指定的质量保存为JPG，以减小文件大小。
        返回新的JPG图片路径。
        """
        try:
            self.log_message("INFO", f"正在压缩图片: {image_path} (质量: {quality}%)")
            img = Image.open(image_path)
            # 构建新的JPG文件名
            directory, filename = os.path.split(image_path)
            name, _ = os.path.splitext(filename)
            new_path = os.path.join(directory, f"{name}_compressed.jpg")
            # 转换为RGB模式（JPG不支持RGBA）并保存
            img.convert('RGB').save(new_path, "JPEG", quality=quality)
            self.log_message("INFO", f"图片已压缩并保存至: {new_path}")
            return new_path
        except Exception as e:
            self.log_message("ERROR", f"压缩图片失败: {e}", exc_info=True)
            return image_path # 压缩失败则返回原图路径

    def start_summary_process(self):
        """启动总结流程的线程"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END) # 清空日志区域
        self.log_text.config(state="disabled")

        selected_api = self.api_service_var.get()
        api_key = self.bailian_api_key_var.get() if selected_api == '阿里云百炼' else self.gemini_api_key_var.get()

        if not api_key:
            messagebox.showwarning("API Key缺失", f"请输入 {selected_api} 的 API Key。")
            return

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stop_event.clear() # 清除停止事件，允许新流程开始

        # --- 新增：隐藏主窗口 ---
        self.withdraw() 

        self.running_process_thread = threading.Thread(target=self.run_summary_process, daemon=True)
        self.running_process_thread.start()

    def stop_summary_process(self):
        """设置停止事件，请求终止当前正在运行的流程"""
        self.log_message("INFO", "收到停止请求...")
        self.stop_event.set()
        self.stop_button.config(state=tk.DISABLED)

    def run_summary_process(self):
        """【核心流程】调用真实的后端模块进行截图、OCR、去重和LLM总结"""
        try:
            # 1. 获取配置
            selected_api = self.api_service_var.get()
            api_key = self.bailian_api_key_var.get() if selected_api == '阿里云百炼' else self.gemini_api_key_var.get()
            
            # 再次检查API Key，以防用户在开始后清空
            if not api_key:
                self.log_message("ERROR", f"API Key缺失: {selected_api} 的 API Key 未提供。")
                messagebox.showwarning("API Key缺失", f"请输入 {selected_api} 的 API Key。")
                return # 提前退出

            max_scrolls = int(self.scrolls_var.get())
            
            # 强制全屏截图，region 始终为 None
            region = None 

            # 2. 截图
            self.log_message("INFO", "准备进行全屏截图...")
            # --- 新增：第一次截图延迟1秒 ---
            time.sleep(1) 
            if self.stop_event.is_set(): raise InterruptedError("用户中止")
            
            image_paths = capture_chat_history_dynamically(
                region=region, # 此时 region 始终为 None，表示全屏
                screenshot_dir="temp_screenshots",
                max_scrolls=max_scrolls
            )
            if self.stop_event.is_set(): raise InterruptedError("用户中止")
            
            # 3. OCR (包含可选的图片压缩)
            self.log_message("INFO", "开始识别文字...")
            processed_image_paths = []
            if self.compress_var.get():
                self.log_message("INFO", "图片压缩功能已启用。")
                for path in image_paths:
                    if self.stop_event.is_set(): raise InterruptedError("用户中止")
                    processed_image_paths.append(self._compress_image(path))
            else:
                processed_image_paths = image_paths # 不压缩，直接使用原图

            all_lines = self.ocr_ai.process_images(processed_image_paths) # 通过OCR适配器处理图片
            if self.stop_event.is_set(): raise InterruptedError("用户中止")

            # 4. 智能去重
            self.log_message("INFO", "开始智能去重...")
            unique_lines = self._deduplicate_messages_smartly(all_lines)
            self.log_message("INFO", f"去重完成，共 {len(unique_lines)} 条独立对话。")
            if self.stop_event.is_set(): raise InterruptedError("用户中止")

            if not unique_lines:
                self.log_message("WARNING", "未识别到有效的聊天记录，无法进行总结。")
                messagebox.showwarning("无聊天记录", "未能从截图中识别到有效的聊天记录，请检查截图区域或尝试重新截图。")
                return

            # 5. LLM总结 (使用LLMManager)
            summary = self.llm_manager.summarize(selected_api, unique_lines, api_key)

            # 6. 显示结果
            self.log_message("INFO", "总结完成！")
            self.after(0, self._display_summary, summary)

        except InterruptedError:
            self.log_message("WARNING", "流程已被用户手动停止。")
        except GUIAutomationError as e:
            self.log_message("ERROR", f"GUI自动化错误: {e}", exc_info=True)
            messagebox.showerror("GUI自动化错误", f"请确保相关应用窗口处于活动状态且可见。\n\n错误详情: {e}")
        except OCRPreprocessingError as e:
            self.log_message("ERROR", f"OCR处理错误: {e}", exc_info=True)
            messagebox.showerror("OCR处理错误", f"图片处理或文字识别失败。\n\n错误详情: {e}")
        except ConnectionError as e:
            self.log_message("ERROR", f"API连接错误: {e}", exc_info=True)
            messagebox.showerror("API连接错误", f"与AI服务连接失败，请检查网络或API Key。\n\n错误详情: {e}")
        except ValueError as e:
            self.log_message("ERROR", f"配置或数据错误: {e}", exc_info=True)
            messagebox.showerror("配置/数据错误", f"配置有误或数据处理异常，请检查输入。\n\n错误详情: {e}")
        except Exception as e:
            self.log_message("ERROR", f"流程发生未知错误: {e}", exc_info=True)
            messagebox.showerror("未知错误", f"发生未知错误:\n\n{e}")
        finally:
            # --- 无论成功、失败还是中止，都重新显示主窗口并启用按钮 ---
            self.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            self.after(0, self.deiconify) # 重新显示主窗口

    def _deduplicate_messages_smartly(self, messages: list, threshold: float = 0.9) -> list:
        """
        智能去重聊天记录，通过比较相似度来判断是否为重复内容。
        """
        if not messages:
            return []
        unique = [messages[0]]
        for msg in messages[1:]:
            # 使用 SequenceMatcher 比较当前消息与上一条唯一消息的相似度
            # 如果相似度低于阈值，则认为是新消息
            if SequenceMatcher(None, unique[-1], msg).ratio() < threshold:
                unique.append(msg)
        return unique

    def _display_summary(self, summary: str):
        """在GUI日志区域显示最终总结结果"""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, "\n\n--- 最终总结结果 ---\n" + summary)
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def on_closing(self):
        """处理窗口关闭事件，提示用户保存配置并退出"""
        if messagebox.askokcancel("退出", "确定要退出吗？"):
            self.stop_event.set() # 确保在退出前停止所有正在运行的线程
            self.save_configuration()
            self.destroy()

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()