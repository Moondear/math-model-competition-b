# GitHub仓库推送脚本
# 使用方法：
# 1. 在GitHub上创建新仓库
# 2. 复制仓库URL
# 3. 运行此脚本并输入URL

Write-Host "=== GitHub仓库推送脚本 ===" -ForegroundColor Green
Write-Host ""

# 检查Git状态
Write-Host "检查Git状态..." -ForegroundColor Yellow
git status

Write-Host ""
Write-Host "请输入您的GitHub仓库URL (例如: https://github.com/username/repo-name.git):" -ForegroundColor Cyan
$githubUrl = Read-Host

if ($githubUrl -eq "") {
    Write-Host "错误：请输入有效的GitHub仓库URL" -ForegroundColor Red
    exit 1
}

# 添加GitHub远程仓库
Write-Host "添加GitHub远程仓库..." -ForegroundColor Yellow
git remote add github $githubUrl

# 检查远程仓库
Write-Host "检查远程仓库配置..." -ForegroundColor Yellow
git remote -v

# 推送代码到GitHub
Write-Host "推送代码到GitHub..." -ForegroundColor Yellow
git push github main

Write-Host ""
Write-Host "完成！代码已推送到GitHub仓库。" -ForegroundColor Green
Write-Host "您可以在GitHub上查看您的代码了。" -ForegroundColor Green 