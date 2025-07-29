#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体工具模块
解决matplotlib中文显示乱码问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import warnings

def setup_chinese_font():
    """设置中文字体，解决乱码问题"""
    
    # 根据操作系统选择合适的中文字体
    system = platform.system()
    
    if system == "Windows":
        # Windows系统字体
        font_candidates = [
            'Microsoft YaHei',      # 微软雅黑
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong',             # 仿宋
            'STXihei',              # 华文黑体
            'STSong',               # 华文宋体
            'STKaiti',              # 华文楷体
            'STFangsong',           # 华文仿宋
        ]
    elif system == "Darwin":  # macOS
        font_candidates = [
            'PingFang SC',          # 苹方
            'Hiragino Sans GB',     # 冬青黑体
            'STHeiti',              # 华文黑体
            'STSong',               # 华文宋体
            'Arial Unicode MS',     # Arial Unicode
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'DejaVu Sans',          # DejaVu Sans
            'Liberation Sans',      # Liberation Sans
            'Arial Unicode MS',     # Arial Unicode
        ]
    
    # 尝试设置字体
    font_found = False
    for font_name in font_candidates:
        try:
            # 检查字体是否可用
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and os.path.exists(font_path):
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                font_found = True
                print(f"✅ 成功设置中文字体: {font_name}")
                break
        except Exception as e:
            continue
    
    if not font_found:
        # 如果找不到中文字体，尝试使用系统默认字体
        try:
            # 获取系统默认字体
            default_font = fm.FontProperties().get_name()
            plt.rcParams['font.sans-serif'] = [default_font]
            print(f"⚠️ 未找到中文字体，使用默认字体: {default_font}")
        except Exception as e:
            print(f"❌ 字体设置失败: {e}")
    
    # 设置负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置字体大小
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16

def get_available_fonts():
    """获取系统中可用的字体列表"""
    fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = []
    
    # 常见中文字体关键词
    chinese_keywords = ['YaHei', 'Hei', 'Song', 'Kai', 'Fang', 'ST', 'PingFang', 'Hiragino', 'WenQuanYi']
    
    for font in fonts:
        for keyword in chinese_keywords:
            if keyword.lower() in font.lower():
                chinese_fonts.append(font)
                break
    
    return chinese_fonts

def test_chinese_display():
    """测试中文字体显示效果"""
    setup_chinese_font()
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试数据
    categories = ['抽样检验', '生产决策', '多工序优化', '鲁棒性分析']
    values = [85, 92, 88, 95]
    
    # 绘制柱状图
    bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%',
                ha='center', va='bottom')
    
    # 设置标题和标签
    ax.set_title('数学建模系统性能测试', fontsize=16, fontweight='bold')
    ax.set_xlabel('功能模块', fontsize=12)
    ax.set_ylabel('性能评分 (%)', fontsize=12)
    
    # 设置y轴范围
    ax.set_ylim(0, 100)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存测试图片
    plt.savefig('output/chinese_font_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 中文字体测试完成，请查看 output/chinese_font_test.png")

def ensure_output_dir():
    """确保输出目录存在"""
    if not os.path.exists('output'):
        os.makedirs('output')

if __name__ == "__main__":
    # 测试中文字体
    test_chinese_display() 