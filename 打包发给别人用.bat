@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "OUTDIR=%~dp0XPS_UPS_Release"
set "ZIP=%~dp0XPS_UPS_Release.zip"

echo ========================================
echo   打包发布包（对方已安装 Python）
echo ========================================
echo.

if exist "%OUTDIR%" rmdir /s /q "%OUTDIR%"
mkdir "%OUTDIR%"

echo [1/3] 复制必需文件...
copy /y "XPS_UPS.py" "%OUTDIR%\" >nul
copy /y "app.py" "%OUTDIR%\" >nul
copy /y "xps_app.py" "%OUTDIR%\" >nul
copy /y "mode_select.py" "%OUTDIR%\" >nul
copy /y "fit_window.py" "%OUTDIR%\" >nul
copy /y "peak_fit.py" "%OUTDIR%\" >nul
copy /y "plots.py" "%OUTDIR%\" >nul
copy /y "reader.py" "%OUTDIR%\" >nul
copy /y "file_loader.py" "%OUTDIR%\" >nul
copy /y "export_csv.py" "%OUTDIR%\" >nul
copy /y "ui_theme.py" "%OUTDIR%\" >nul
copy /y "xps_chem_states.py" "%OUTDIR%\" >nul
copy /y "autofit_engine.py" "%OUTDIR%\" >nul
if exist "main_app.py" copy /y "main_app.py" "%OUTDIR%\" >nul
copy /y "requirements.txt" "%OUTDIR%\" >nul
copy /y "运行.bat" "%OUTDIR%\" >nul
if exist "README.md" copy /y "README.md" "%OUTDIR%\" >nul

echo [1b/3] 复制 DL 模型 (5 个 *.pt, ~38 MB)...
if exist "models" (
    xcopy /s /e /y /i "models" "%OUTDIR%\models\" >nul
) else (
    echo   [警告] models\ 目录不存在，发布包将不带 DL 模型，自动拟合不可用
)

echo [2/3] 生成对方使用说明...
set "README_TXT=%OUTDIR%\使用说明.txt"
powershell -NoProfile -Command ^
  "$p = '%README_TXT%';" ^
  "$lines = @(" ^
  "  '运行方法（对方电脑已安装 Python）：'," ^
  "  ''," ^
  "  '1. 解压后，双击 运行.bat'," ^
  "  ''," ^
  "  '2. 若首次运行缺库，在此文件夹打开终端执行：'," ^
  "  '   pip install -r requirements.txt'," ^
  "  ''," ^
  "  '3. 也可直接运行：'," ^
  "  '   python XPS_UPS.py'," ^
  "  ''," ^
  "  '说明：'," ^
  "  '- 启动后先选择 UPS 或 XPS 模块。'," ^
  "  '- UPS 读取 .ibw 需要 igor2；建议用 Python 3.11/3.10。'" ^
  ");" ^
  "$lines | Out-File -FilePath $p -Encoding UTF8"

echo [3/3] 压缩为 zip...
powershell -NoProfile -Command "Compress-Archive -Path \"%OUTDIR%\\*\" -DestinationPath \"%ZIP%\" -Force"
if errorlevel 1 (
  echo.
  echo [错误] 压缩失败。你仍可以直接把 %OUTDIR% 文件夹发给对方。
  pause
  exit /b 1
)

echo.
echo 完成：
echo - 发布文件夹：%OUTDIR%
echo - 发布压缩包：%ZIP%
echo.
explorer "%~dp0"
pause

