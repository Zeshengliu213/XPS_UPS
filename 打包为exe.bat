@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ========================================
echo   打包独立 exe（对方无需安装 Python）
echo ========================================
echo.

set "PY=py -3.13"
%PY% --version >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 Python 3.13，请先安装 Python 3.13 ^(或修改本脚本中的 PY 变量^)。
  pause
  exit /b 1
)

echo [1/4] 确认依赖...
%PY% -m pip install --quiet --upgrade pyinstaller
%PY% -m pip install --quiet -r requirements.txt construct

echo [2/4] 清理旧构建...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist XPS_UPS.spec del /q XPS_UPS.spec

echo [3/4] PyInstaller 打包中（几分钟，请耐心等待）...
%PY% -m PyInstaller ^
  --noconfirm ^
  --onefile ^
  --windowed ^
  --name XPS_UPS ^
  --collect-all igor2 ^
  --collect-all construct ^
  --collect-submodules matplotlib ^
  --collect-submodules scipy ^
  --hidden-import tkinter ^
  --hidden-import tkinter.ttk ^
  --hidden-import tkinter.filedialog ^
  --hidden-import tkinter.messagebox ^
  XPS_UPS.py

if errorlevel 1 (
  echo.
  echo [错误] 打包失败，请查看上方日志。
  pause
  exit /b 1
)

echo [4/4] 打包 zip 方便分发...
set "OUT=%~dp0XPS_UPS_exe"
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%"
copy /y "dist\XPS_UPS.exe" "%OUT%\" >nul
if exist README.md copy /y "README.md" "%OUT%\" >nul

powershell -NoProfile -Command ^
  "$p = '%OUT%\使用说明.txt';" ^
  "$lines = @(" ^
  "  'XPS / UPS IBW Processor - 独立可执行版'," ^
  "  ''," ^
  "  '使用方法：双击 XPS_UPS.exe 即可运行，无需安装 Python。'," ^
  "  ''," ^
  "  '首次启动可能较慢（几秒到十几秒），请耐心等待。'," ^
  "  ''," ^
  "  '若 Windows Defender 提示，请选择“仍要运行”。'" ^
  ");" ^
  "$lines | Out-File -FilePath $p -Encoding UTF8"

set "ZIP=%~dp0XPS_UPS_exe.zip"
if exist "%ZIP%" del /q "%ZIP%"
powershell -NoProfile -Command "Compress-Archive -Path \"%OUT%\\*\" -DestinationPath \"%ZIP%\" -Force"

echo.
echo ========================================
echo 完成！
echo - exe 文件：%OUT%\XPS_UPS.exe
echo - 发布压缩包：%ZIP%
echo ========================================
explorer "%~dp0"
pause
