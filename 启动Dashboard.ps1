# Dashboard启动脚本
# 2024数学建模智能决策系统

Write-Host "🎯 启动数学建模智能决策系统..." -ForegroundColor Green
Write-Host ""

# 检查当前目录
$currentDir = Get-Location
Write-Host "📁 当前目录: $currentDir" -ForegroundColor Yellow

# 检查必要文件
if (Test-Path "dashboard_lite.py") {
    Write-Host "✅ 找到轻量版Dashboard文件" -ForegroundColor Green
} else {
    Write-Host "❌ 未找到dashboard_lite.py文件" -ForegroundColor Red
    exit 1
}

# 检查端口占用情况
Write-Host "🔍 检查端口状态..." -ForegroundColor Cyan
$port8501 = netstat -an | findstr ":8501"
$port8502 = netstat -an | findstr ":8502"

if ($port8501) {
    Write-Host "📌 端口8501已被占用（可能是原版Dashboard）" -ForegroundColor Yellow
}

if ($port8502) {
    Write-Host "📌 端口8502已被占用（轻量版Dashboard运行中）" -ForegroundColor Yellow
    Write-Host "🌐 轻量版Dashboard已运行: http://localhost:8502" -ForegroundColor Green
} else {
    Write-Host "🚀 启动轻量版Dashboard..." -ForegroundColor Cyan
    streamlit run dashboard_lite.py --server.port 8502
}

Write-Host ""
Write-Host "📖 使用说明:" -ForegroundColor Magenta
Write-Host "   🔗 浏览器访问: http://localhost:8502" -ForegroundColor White
Write-Host "   ⭐ 特色功能: 完全本地运行，无需外部依赖库" -ForegroundColor White
Write-Host "   🎛️ 包含功能: 抽样检验、生产决策、多工序优化、鲁棒分析" -ForegroundColor White
Write-Host "" 