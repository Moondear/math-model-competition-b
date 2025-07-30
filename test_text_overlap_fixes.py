#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文字重叠问题修复效果
"""

import sys
import os
sys.path.append('src')

from competition_b_solver import CompetitionBSolver
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_overlap_fixes():
    """测试文字重叠问题的修复效果"""
    print("开始测试文字重叠问题修复效果...")
    
    # 创建求解器
    solver = CompetitionBSolver()
    
    # 测试问题二的可视化修复
    print("\n=== 测试问题二可视化修复 ===")
    for case_id in range(1, 4):  # 测试前3个情况
        print(f"\n正在测试情况 {case_id}...")
        
        try:
            # 求解问题二
            result = solver.solve_problem2_production(case_id)
            
            # 打印结果
            print(f"情况 {case_id} 结果:")
            print(f"  决策: {result['decisions']}")
            print(f"  期望利润: {result['expected_profit']:.2f}")
            print(f"  图像路径: {result['plot_path']}")
            
            # 检查图像文件是否存在
            if os.path.exists(result['plot_path']):
                file_size = os.path.getsize(result['plot_path'])
                print(f"  图像文件大小: {file_size} 字节")
            else:
                print("  警告: 图像文件未生成!")
                
        except Exception as e:
            print(f"  错误: {e}")
    
    # 测试问题三的可视化修复
    print("\n=== 测试问题三可视化修复 ===")
    try:
        # 求解问题三
        result = solver.solve_problem3_multistage()
        
        # 打印结果
        print(f"问题三结果:")
        print(f"  决策: {result['decisions']}")
        print(f"  总成本: {result['total_cost']:.2f}")
        print(f"  图像路径: {result['plot_path']}")
        
        # 检查图像文件是否存在
        if os.path.exists(result['plot_path']):
            file_size = os.path.getsize(result['plot_path'])
            print(f"  图像文件大小: {file_size} 字节")
        else:
            print("  警告: 图像文件未生成!")
            
    except Exception as e:
        print(f"  错误: {e}")
    
    print("\n测试完成!")
    print("请检查 output/ 目录中的图像文件:")
    print("- 问题二: 确认决策流程图中的文字有白色背景框，不与连接线重叠")
    print("- 问题三: 确认图例位置在右上角，不与网络节点重叠")
    print("- 确认所有文字都清晰可读，没有重叠问题")

if __name__ == "__main__":
    test_text_overlap_fixes() 