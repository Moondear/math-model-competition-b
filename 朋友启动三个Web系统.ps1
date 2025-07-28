# 🎯 数学建模项目 - 三大Web系统启动器
# PowerShell版本，适合Windows用户

Write-Host "🎯 数学建模项目 - 三大Web系统启动器" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 检查Python环境
Write-Host "📦 检查Python环境..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python环境正常: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python未安装，请先安装Python" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit
}

# 安装依赖
Write-Host "📦 安装必要依赖包..." -ForegroundColor Yellow
pip install streamlit plotly pandas numpy scipy matplotlib -q
Write-Host "✅ 依赖安装完成" -ForegroundColor Green

Write-Host ""
Write-Host "🚀 启动三大Web系统..." -ForegroundColor Cyan
Write-Host ""

# 启动三个Web服务
Write-Host "📊 启动智能决策仪表盘 (端口8510)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-Command", "streamlit run dashboard_safe.py --server.port 8510" -WindowStyle Minimized
Start-Sleep 3

Write-Host "🎮 启动沉浸式VR/AR展示 (端口8503)..." -ForegroundColor Blue  
Start-Process powershell -ArgumentList "-Command", "streamlit run interactive_showcase.py --server.port 8503" -WindowStyle Minimized
Start-Sleep 3

Write-Host "🤖 启动AI答辩教练系统 (端口8505)..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-Command", "streamlit run ai_defense_system.py --server.port 8505" -WindowStyle Minimized
Start-Sleep 3

Write-Host ""
Write-Host "🎊 所有系统启动完成！" -ForegroundColor Green
Write-Host ""
Write-Host "📍 访问地址：" -ForegroundColor Yellow
Write-Host "   📊 智能决策仪表盘：http://localhost:8510" -ForegroundColor White
Write-Host "   🎮 沉浸式VR/AR展示：http://localhost:8503" -ForegroundColor White
Write-Host "   🤖 AI答辩教练系统：http://localhost:8505" -ForegroundColor White
Write-Host ""
Write-Host "💡 提示：浏览器即将自动打开，请等待几秒让服务完全启动" -ForegroundColor Yellow
Write-Host ""

# 等待服务启动并打开浏览器
Start-Sleep 5
Start-Process "http://localhost:8510"
Start-Sleep 2
Start-Process "http://localhost:8503"
Start-Sleep 2  
Start-Process "http://localhost:8505"

Write-Host "🎉 享受您的数学建模之旅！" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  注意：关闭此窗口将停止所有Web服务" -ForegroundColor Red
Read-Host "按Enter键退出" 