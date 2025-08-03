# GitHub仓库导入指南

## 方法1：使用自动化脚本（推荐）

1. **在GitHub上创建新仓库**：
   - 访问 https://github.com
   - 点击右上角的 "+" 号，选择 "New repository"
   - 输入仓库名称（建议：`math-model-competition-b`）
   - 选择 "Public" 或 "Private"
   - **不要**勾选 "Initialize this repository with a README"
   - 点击 "Create repository"

2. **复制仓库URL**：
   - 创建完成后，复制仓库的URL
   - 格式类似：`https://github.com/yourusername/math-model-competition-b.git`

3. **运行推送脚本**：
   ```powershell
   .\push_to_github.ps1
   ```
   - 按提示输入GitHub仓库URL
   - 脚本会自动完成推送

## 方法2：手动操作

### 步骤1：在GitHub上创建仓库
- 访问 https://github.com
- 创建新仓库（不要初始化README）

### 步骤2：添加GitHub远程仓库
```powershell
# 替换为您的GitHub仓库URL
git remote add github https://github.com/yourusername/repo-name.git
```

### 步骤3：推送代码
```powershell
git push github main
```

## 方法3：使用GitHub Desktop

1. 下载并安装 GitHub Desktop
2. 添加本地仓库
3. 发布到GitHub

## 验证推送结果

推送完成后，您可以：
1. 访问您的GitHub仓库页面
2. 确认所有文件都已上传
3. 检查README.md是否正确显示

## 常见问题解决

### 网络连接问题
如果遇到SSL/TLS连接失败：
```powershell
git config --global http.sslVerify false
```

### 认证问题
如果遇到认证失败，请确保：
1. 在GitHub上设置了SSH密钥，或
2. 使用个人访问令牌进行HTTPS认证

## 仓库内容说明

您的仓库包含：
- 数学建模竞赛B题的完整解决方案
- Python代码实现
- 可视化图表和报告
- 答辩准备材料
- 详细的算法说明

## 后续维护

推送完成后，您可以：
1. 在GitHub上添加项目描述
2. 设置合适的标签
3. 添加项目截图
4. 更新README.md文件 