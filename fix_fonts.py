#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复中文字体设置脚本
"""

import os
import re

def fix_font_settings():
    """修复所有文件中的中文字体设置"""
    
    # 需要修复的文件列表
    files_to_fix = [
        'src/visualization.py',
        'quick_demo.py'
    ]
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            continue
            
        print(f"正在修复 {file_path}...")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换字体设置
        # 1. 替换 plt.rcParams['font.sans-serif'] = ['SimHei'] 为 setup_chinese_font()
        content = re.sub(
            r"plt\.rcParams\['font\.sans-serif'\] = \['SimHei'\]\s*\n\s*plt\.rcParams\['axes\.unicode_minus'\] = False",
            "setup_chinese_font()",
            content
        )
        
        # 2. 替换 plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS'] 为 setup_chinese_font()
        content = re.sub(
            r"plt\.rcParams\['font\.sans-serif'\] = \['Microsoft YaHei', 'SimHei', 'Arial Unicode MS'\]",
            "setup_chinese_font()",
            content
        )
        
        # 3. 确保导入了font_utils
        if 'from .font_utils import setup_chinese_font' not in content and 'from font_utils import setup_chinese_font' not in content:
            # 在import语句后添加
            if 'import matplotlib.pyplot as plt' in content:
                content = content.replace(
                    'import matplotlib.pyplot as plt',
                    'import matplotlib.pyplot as plt\nfrom .font_utils import setup_chinese_font, ensure_output_dir'
                )
        
        # 写入修复后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ {file_path} 修复完成")

if __name__ == "__main__":
    fix_font_settings() 