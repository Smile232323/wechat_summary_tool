# 微信聊天记录自动化总结工具

这是一个基于 Python 的桌面应用，旨在帮助用户自动化截取微信聊天记录，并通过大语言模型（LLM）对其进行智能总结，提取关键信息、决策点和待办事项。

## ✨ 功能特性

-   **全屏自动化截图：** 自动识别微信窗口并进行全屏截图，模拟滚动以捕获完整的聊天记录。
-   **智能图片去重：** 基于图像相似度（SSIM）算法，在滚动截图过程中自动识别并跳过高度重复的屏幕内容（相似度达到 80% 以上即停止），避免无效截图。
-   **OCR 文字识别：** 将截图中的图片内容转换为可编辑的文本。
-   **智能文本去重：** 对 OCR 识别出的聊天文本进行智能去重，确保总结内容的唯一性。
-   **多LLM支持：** 支持阿里云百炼和 Google Gemini 两种大语言模型进行聊天内容总结。
-   **配置保存：** 用户配置（API Key、滚动次数等）会自动保存，方便下次使用。
-   **GUI 界面：** 提供直观的图形用户界面，操作简便。
-   **后台运行：** 开始总结后，主界面自动隐藏，任务完成后自动显示。
-   **首次截图延迟：** 首次截图前有 1 秒延迟，确保微信窗口准备就绪。

## 🚀 如何使用

### 1. 从源代码运行

#### 前提条件

-   **Python 3.8+**: 推荐使用 [Python 官方网站](https://www.python.org/downloads/) 下载安装。
-   **Tesseract OCR 引擎**: OCR 模块依赖于 Tesseract。您需要从 [Tesseract GitHub](https://tesseract-ocr.github.io/tessdoc/Downloads.html) 下载并安装它。
    -   **重要**: 安装时请确保勾选“Add to PATH”选项，或者手动将其安装路径（例如 `C:\Program Files\Tesseract-OCR`）添加到系统环境变量 `PATH` 中。
-   **微信PC客户端**: 确保您的电脑上安装并登录了微信PC客户端。

#### 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/YourGitHubUsername/wechat_summary_tool.git
    cd wechat_summary_tool
    ```
2.  **创建并激活虚拟环境 (推荐):**
    ```bash
    python -m venv .venv
    # Windows (在命令提示符或PowerShell中)
    .\.venv\Scripts\activate
    # Windows (在Git Bash中)
    source .venv/Scripts/activate
    ```
3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **配置 API Keys:**
    -   将 `config.json.example` 文件复制并重命名为 `config.json`。
    -   打开 `config.json`，在 `bailian_api_key` 和 `gemini_api_key` 字段中填入您的阿里云百炼 API Key 和 Google Gemini API Key。
    -   **注意：** 您的 `config.json` 文件可能包含 `llm_api_key`、`chat_area_coords` 和 `fullscreen` 等字段。这些是旧版本遗留的配置，当前版本程序已不再使用它们来控制截图区域（程序强制全屏截图），但请保留它们以确保配置文件格式兼容性。您只需关注 `bailian_api_key` 和 `gemini_api_key` 的填写。
    -   您可以从 [阿里云百炼控制台](https://www.aliyun.com/product/bailian) 和 [Google AI Studio](https://aistudio.google.com/app/apikey) 获取您的 API Key。
    -   **注意:** 为了安全，`config.json` 已被 `.gitignore` 忽略，不会上传到 GitHub。
5.  **运行应用程序:**
    ```bash
    python main_app.py
    ```

### 2. 使用可执行文件 (EXE)

**注意：** 即使使用可执行文件，**Tesseract OCR 引擎** 仍需要单独安装在您的电脑上，并确保其路径已添加到系统环境变量 `PATH` 中。

1.  **下载可执行文件:**
    -   访问项目的 [Releases 页面](https://github.com/YourGitHubUsername/wechat_summary_tool/releases) (待您发布后可见)。
    -   下载最新版本的 `wechat_summary_tool.exe`。
2.  **配置 API Keys:**
    -   下载的可执行文件通常会附带一个 `config.json` 文件（或您需要手动创建）。
    -   请确保 `config.json` 文件与 `.exe` 文件在同一目录下。
    -   按照上述“从源代码运行”中的步骤配置 `config.json`。
3.  **运行:**
    -   双击 `wechat_summary_tool.exe` 即可运行。

## 📦 项目结构
wechat_summary_tool/
├── .gitignore             # Git 忽略文件配置
├── README.md              # 项目说明文档
├── requirements.txt       # Python 依赖列表
├── config.json.example    # 配置文件示例 (用户需复制为 config.json)
├── main_app.py            # 应用程序主入口和核心逻辑
├── .venv/                 # Python 虚拟环境 (被 .gitignore 忽略)
├── src/                   # 核心源代码目录
│   ├── init.py
│   ├── gui_automation/    # GUI 自动化相关模块
│   │   ├── init.py
│   │   └── wechat_gui_automator.py # 微信窗口查找、截图、滚动逻辑
│   ├── llm_integration/   # 大语言模型集成模块 (保留目录结构，但LLMManager类在main_app.py中)
│   │   └── init.py
│   ├── main_app/          # 主应用相关模块 (保留目录结构，但GUI逻辑在main_app.py中)
│   │   └── init.py
│   └── ocr_processing/    # OCR 处理模块
│       ├── init.py
│       └── ocr_processor.py # 图像文字识别和文本清洗
├── dist/                  # PyInstaller 打包输出目录 (被 .gitignore 忽略)
├── build/                 # PyInstaller 临时构建目录 (被 .gitignore 忽略)
├── temp_screenshots/      # 临时截图保存目录 (被 .gitignore 忽略)
└── wechat_summary_tool.log # 应用程序日志文件 (被 .gitignore 忽略)

## 🛠️ 打包为可执行文件 (开发者)

如果您想自己打包 `.exe` 文件，请遵循以下步骤：

1.  **确保所有依赖已安装** (`pip install -r requirements.txt`)。
2.  **安装 PyInstaller:**
    ```bash
    pip install pyinstaller
    ```
3.  **执行打包命令:**
    在项目根目录（`wechat_summary_tool` 文件夹，即 `main_app.py` 所在的目录）下运行：
    ```bash
    pyinstaller --noconfirm --onefile --windowed \
    --add-data "config.json.example;." \
    --add-data "src;src" \
    --hidden-import "win32timezone" \
    --hidden-import "pyautogui" \
    --hidden-import "pygetwindow" \
    --hidden-import "pyscreeze" \
    --hidden-import "pytesseract" \
    --name "wechat_summary_tool" main_app.py
    ```
    -   `--noconfirm`: 不询问直接覆盖旧文件。
    -   `--onefile`: 打包成一个单独的 `.exe` 文件。
    -   `--windowed` 或 `--noconsole`: 运行 `.exe` 时不显示命令行窗口（适用于 GUI 应用）。
    -   `--add-data "config.json.example;."`: 将 `config.json.example` 文件添加到 `.exe` 所在目录。
    -   `--add-data "src;src"`: 将 `src` 文件夹及其所有内容添加到 `.exe` 的内部结构中，确保模块导入正确。
    -   `--hidden-import "..."`: 这些参数用于显式包含 PyInstaller 可能无法自动检测到的模块，确保 `.exe` 运行时不会缺少依赖。
    -   `--name "wechat_summary_tool"`: 指定生成的可执行文件名。
    -   `main_app.py`: 您的主入口文件。

    打包成功后，您将在 `dist/` 目录下找到 `wechat_summary_tool.exe`。

## 🐛 故障排除

-   **`AttributeError: module 'cv2' has no attribute 'compareSSIM'`**: 这通常意味着 `opencv-python` 安装不完整。请尝试卸载 `opencv-python` 并安装 `opencv-contrib-python`：
    `pip uninstall opencv-python`
    `pip install opencv-contrib-python`
-   **`GUIAutomationError: 未找到微信窗口`**:
    -   确保微信PC客户端已打开且未最小化。
    -   尝试将微信窗口置于前台。
    -   如果问题依旧，可能是您的微信窗口标题或类名与程序中预设的不符。您可以尝试在 `src/gui_automation/wechat_gui_automator.py` 的 `find_wechat_window` 函数中添加或修改 `possible_titles_parts` 列表。
-   **OCR 识别失败或不准确**:
    -   检查 Tesseract OCR 引擎是否已正确安装，并且其安装路径已添加到系统 `PATH` 环境变量中。
    -   尝试更新 `pytesseract` 和 `opencv-contrib-python` 到最新版本。
    -   确保截图清晰，无遮挡。
-   **LLM 总结失败 (ConnectionError)**:
    -   检查您的 API Key 是否正确填写在 `config.json` 中。
    -   检查您的网络连接。
    -   确认您选择的 LLM 服务（阿里云百炼或 Google Gemini）在您所在地区可用，且您的账户有足够的配额。

---

## 🤝 贡献

欢迎提出问题 (Issues) 或提交拉取请求 (Pull Requests) 来改进此项目。

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 发布。