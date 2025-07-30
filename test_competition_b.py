#!/usr/bin/env python3
"""
测试2024年数学建模竞赛B题求解器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_competition_b_solver():
    """测试竞赛B题求解器"""
    print("🧪 开始测试2024年数学建模竞赛B题求解器...")
    print("=" * 60)
    
    try:
        from src.competition_b_solver import CompetitionBSolver
        
        # 创建求解器实例
        solver = CompetitionBSolver()
        print("✅ 求解器初始化成功")
        
        # 测试问题1：抽样检验
        print("\n🔍 测试问题1：抽样检验方案设计...")
        try:
            result1 = solver.solve_problem1_sampling()
            print(f"✅ 问题1求解成功")
            print(f"   样本量: {result1['sample_size']}")
            print(f"   判定值: {result1['acceptance_number']}")
            print(f"   实际α: {result1['actual_alpha']:.4f}")
            print(f"   实际β: {result1['actual_beta']:.4f}")
        except Exception as e:
            print(f"❌ 问题1求解失败: {e}")
        
        # 测试问题2：生产决策（测试情况1）
        print("\n🏭 测试问题2：生产流程决策（情况1）...")
        try:
            result2 = solver.solve_problem2_production(1)
            print(f"✅ 问题2求解成功")
            print(f"   检测零件1: {'是' if result2['decisions']['test_component1'] else '否'}")
            print(f"   检测零件2: {'是' if result2['decisions']['test_component2'] else '否'}")
            print(f"   检测成品: {'是' if result2['decisions']['test_finished_product'] else '否'}")
            print(f"   拆解不合格品: {'是' if result2['decisions']['disassemble_defective'] else '否'}")
            print(f"   期望利润: {result2['expected_profit']:.2f}")
        except Exception as e:
            print(f"❌ 问题2求解失败: {e}")
        
        # 测试问题3：多工序优化
        print("\n🌐 测试问题3：多工序网络优化...")
        try:
            result3 = solver.solve_problem3_multistage()
            print(f"✅ 问题3求解成功")
            print(f"   网络规模: {result3['network_size']}个节点")
            print(f"   总成本: {result3['total_cost']:.2f}")
        except Exception as e:
            print(f"❌ 问题3求解失败: {e}")
        
        # 测试问题4：鲁棒优化
        print("\n🛡️ 测试问题4：鲁棒优化分析...")
        try:
            result4 = solver.solve_problem4_uncertainty()
            print(f"✅ 问题4求解成功")
            print(f"   最坏情况利润: {result4['worst_case_profit']:.2f}")
        except Exception as e:
            print(f"❌ 问题4求解失败: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("📊 检查output/文件夹中的结果文件:")
        print("   📄 综合报告: competition_b_comprehensive_report.txt")
        print("   📈 可视化图表: problem1_sampling_analysis.png")
        print("   📈 可视化图表: problem2_case1_decision_tree.png")
        print("   📈 可视化图表: problem3_multistage_network.png")
        print("   📈 可视化图表: problem4_uncertainty_analysis.png")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("💡 请确保已安装所需依赖: pip install numpy scipy matplotlib plotly networkx")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_competition_b_solver() 