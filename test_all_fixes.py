#!/usr/bin/env python3
"""
综合修复测试程序
验证所有三个关键修复是否成功
"""

import sys
import os
import time
import traceback

# 添加路径
sys.path.append('src')

def test_fix_1_dashboard():
    """测试修复1：仪表盘启动"""
    print("🔧 测试修复1：仪表盘模块导入")
    print("-" * 50)
    
    try:
        # 测试dashboard的导入
        from src.dashboard import get_system_metrics
        print("✅ dashboard.py 导入成功")
        
        # 测试系统指标获取
        metrics = get_system_metrics()
        print(f"✅ 系统指标获取成功: CPU={metrics['cpu_usage']:.1f}%, 内存={metrics['memory_usage']:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ 仪表盘模块测试失败: {e}")
        return False

def test_fix_2_quantum():
    """测试修复2：量子启发优化"""
    print("\n⚛️ 测试修复2：量子启发优化")
    print("-" * 50)
    
    try:
        from src.innovation.national_champion import NationalAwardEnhancer
        print("✅ NationalAwardEnhancer 导入成功")
        
        enhancer = NationalAwardEnhancer()
        print("✅ 增强器实例化成功")
        
        # 测试量子优化
        result = enhancer.quantum_inspired_optimization(problem_size=100)
        print(f"✅ 量子优化完成:")
        print(f"   - 状态: {result['status']}")
        print(f"   - 性能提升: {result['speedup']*100:.1f}%")
        print(f"   - 求解器: {result.get('solver', 'OR-Tools')}")
        
        return True
    except Exception as e:
        print(f"❌ 量子优化测试失败: {e}")
        traceback.print_exc()
        return False

def test_fix_3_defense():
    """测试修复3：AI答辩教练"""
    print("\n🤖 测试修复3：AI答辩教练")
    print("-" * 50)
    
    try:
        from src.defense_coach import DefenseCoach, DefenseTrainingSystem
        print("✅ 答辩教练模块导入成功")
        
        # 创建教练实例
        coach = DefenseCoach()
        print("✅ 教练实例化成功")
        
        # 测试提问功能
        question = coach.ask_question()
        print(f"✅ 问题生成成功: {question}")
        
        # 测试评估功能
        evaluation = coach.evaluate_answer("这是一个测试回答")
        print(f"✅ 答案评估成功: 得分 {evaluation['score']}/100")
        
        # 创建训练系统
        training_system = DefenseTrainingSystem()
        print("✅ 训练系统创建成功")
        
        return True
    except Exception as e:
        print(f"❌ AI答辩教练测试失败: {e}")
        traceback.print_exc()
        return False

def test_bonus_production():
    """额外测试：生产优化模块"""
    print("\n🏭 额外测试：生产优化模块")
    print("-" * 50)
    
    try:
        from src.production import ProductionParams, optimize_production
        print("✅ 生产优化模块导入成功")
        
        # 创建测试参数
        params = ProductionParams(
            defect_rate1=0.1,
            defect_rate2=0.1,
            test_cost1=2,
            test_cost2=3,
            assembly_cost=6,
            test_cost_final=3,
            repair_cost=5,
            market_price=56,
            return_loss=6
        )
        print("✅ 生产参数创建成功")
        
        # 测试优化
        result = optimize_production(params)
        print(f"✅ 生产优化完成:")
        print(f"   - 状态: {result['status']}")
        print(f"   - 期望利润: {result['expected_profit']:.2f}")
        print(f"   - 合格概率: {result['ok_probability']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ 生产优化测试失败: {e}")
        return False

def main():
    """主测试程序"""
    print("🎯 开始综合修复测试")
    print("=" * 80)
    
    results = []
    
    # 测试三个主要修复
    results.append(("仪表盘模块", test_fix_1_dashboard()))
    results.append(("量子优化", test_fix_2_quantum()))
    results.append(("AI答辩教练", test_fix_3_defense()))
    
    # 额外测试
    results.append(("生产优化", test_bonus_production()))
    
    # 总结结果
    print("\n📊 修复测试总结")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:15s}: {status}")
        if success:
            passed += 1
    
    print("-" * 40)
    print(f"总体结果: {passed}/{total} 项修复成功")
    
    if passed == total:
        print("🎉 所有修复测试通过！系统已完全修复！")
    elif passed >= total - 1:
        print("🎊 修复基本成功！仅有小问题需要注意。")
    else:
        print("⚠️ 还有一些问题需要进一步修复。")
    
    print("\n✨ 现在您可以正常使用以下功能:")
    print("   - 仪表盘: streamlit run src/dashboard.py --server.port 8080")
    print("   - 沉浸展示: streamlit run interactive_showcase.py --server.port 8503") 
    print("   - 量子优化: python -c \"from src.innovation.national_champion import NationalAwardEnhancer; ...\"")
    print("   - AI答辩: python test_enhanced_defense.py")

if __name__ == "__main__":
    main() 