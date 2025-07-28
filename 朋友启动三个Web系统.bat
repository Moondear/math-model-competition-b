@echo off
echo 🎯 数学建模项目 - 三大Web系统启动器
echo ==========================================
echo.

echo 📦 检查Python环境...
python --version
if errorlevel 1 (
    echo ❌ Python未安装，请先安装Python
    pause
    exit
)

echo ✅ Python环境正常

echo 📦 安装依赖包...
pip install streamlit plotly pandas numpy -q

echo 🚀 启动三大Web系统...
echo.

echo 📊 启动智能决策仪表盘 (端口8510)...
start /min cmd /c "streamlit run dashboard_safe.py --server.port 8510"
timeout 3

echo 🎮 启动沉浸式VR/AR展示 (端口8503)...  
start /min cmd /c "streamlit run interactive_showcase.py --server.port 8503"
timeout 3

echo 🤖 启动AI答辩教练系统 (端口8505)...
start /min cmd /c "streamlit run ai_defense_system.py --server.port 8505"
timeout 3

echo.
echo 🎊 所有系统启动完成！
echo.
echo 📍 访问地址：
echo    📊 智能决策仪表盘：http://localhost:8510
echo    🎮 沉浸式VR/AR展示：http://localhost:8503
echo    🤖 AI答辩教练系统：http://localhost:8505
echo.
echo 💡 提示：浏览器会自动打开，请等待几秒让服务完全启动
echo.

timeout 5
start http://localhost:8510
timeout 2
start http://localhost:8503  
timeout 2
start http://localhost:8505

echo 🎉 享受您的数学建模之旅！
echo 按任意键退出...
pause 