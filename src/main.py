"""
主程序入口
"""
from .sampling import optimal_sampling, run_stress_test
from .production import ProductionParams, optimize_production
from .multistage import create_example_network, optimize_multistage
from .robust import (UncertaintyParams, robust_optimize_production,
                       robust_optimize_multistage)
from .visualization import (create_sampling_dashboard, 
                             create_production_dashboard,
                             create_multistage_dashboard,
                             create_robust_dashboard)
from .latex_generator import generate_paper, save_paper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sampling():
    """分析抽样检验方案"""
    logger.info("开始分析抽样检验方案...")
    
    try:
        # 计算最优方案
        n, c, alpha, beta = optimal_sampling()
        print(f"\n最优抽样方案:")
        print(f"样本量 n = {n}")
        print(f"判定值 c = {c}")
        print(f"实际 α = {alpha:.4f}")
        print(f"实际 β = {beta:.4f}")
        
        # 创建可视化
        print("\n正在生成可视化图表...")
        dist_path, error_path = create_sampling_dashboard(n, c)
        print(f"图表已保存:")
        print(f"1. 概率分布图: {dist_path}")
        print(f"2. 错误率曲线: {error_path}")
        
        # 运行压力测试
        print("\n开始运行压力测试...")
        run_stress_test(n_iterations=100)  # 减少迭代次数，避免并行计算问题
        
    except Exception as e:
        logger.error(f"抽样方案分析失败: {str(e)}")

def analyze_production():
    """分析生产决策优化"""
    logger.info("开始分析生产决策优化...")
    
    try:
        # 基准情况
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
        
        # 优化决策
        result = optimize_production(params)
        
        print("\n生产决策优化结果:")
        print(f"检测零件1: {'是' if result['test_part1'] else '否'}")
        print(f"检测零件2: {'是' if result['test_part2'] else '否'}")
        print(f"检测成品: {'是' if result['test_final'] else '否'}")
        print(f"不合格时拆解: {'是' if result['repair'] else '否'}")
        print(f"期望利润: {result['expected_profit']:.2f}")
        print(f"合格率: {result['p_ok']:.2%}")
        print(f"求解状态: {result['solver_status']}")
        print(f"求解时间: {result['solution_time']*1000:.2f}ms")
        
        # 创建可视化
        print("\n正在生成可视化图表...")
        decision_path, cost_path = create_production_dashboard(result, params)
        print(f"图表已保存:")
        print(f"1. 决策流程图: {decision_path}")
        print(f"2. 成本收益分析: {cost_path}")
        
    except Exception as e:
        logger.error(f"生产决策分析失败: {str(e)}")

def analyze_multistage():
    """分析多工序扩展"""
    logger.info("开始分析多工序扩展...")
    
    try:
        # 创建示例网络
        graph = create_example_network()
        
        # 运行优化
        result = optimize_multistage(graph)
        
        print("\n多工序优化结果:")
        print(f"总成本: {result['total_cost']:.2f}")
        print(f"求解状态: {result['solver_status']}")
        print(f"求解时间: {result['solution_time']*1000:.2f}ms")
        
        print("\n各节点决策:")
        for node, decision in result['decisions'].items():
            print(f"\n节点 {node}:")
            print(f"  检测: {'是' if decision['test'] else '否'}")
            print(f"  返修: {'是' if decision['repair'] else '否'}")
            print(f"  合格率: {decision['p_ok']:.2%}")
            print(f"  成本: {decision['cost']:.2f}")
            
        # 创建可视化
        print("\n正在生成可视化图表...")
        network_path, cost_path = create_multistage_dashboard(graph, result)
        print(f"图表已保存:")
        print(f"1. 生产网络图: {network_path}")
        print(f"2. 成本分布图: {cost_path}")
        
    except Exception as e:
        logger.error(f"多工序优化分析失败: {str(e)}")

def analyze_robust():
    """分析鲁棒优化"""
    logger.info("开始分析鲁棒优化...")
    
    try:
        # 设置不确定性参数（使用较小的样本量）
        uncertainty_params = UncertaintyParams(
            n_samples=50,      # 减少抽样数量
            n_simulations=50,  # 减少模拟次数
            confidence_level=0.95
        )
        
        # 测试生产决策鲁棒优化
        print("\n测试生产决策鲁棒优化...")
        base_params = ProductionParams(
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
        
        result = robust_optimize_production(base_params, uncertainty_params)
        
        print(f"鲁棒决策:")
        for key, value in result['robust_decision'].items():
            print(f"  {key}: {'是' if value else '否'}")
        print(f"期望利润: {result['expected_profit']:.2f}")
        print(f"最差情况利润: {result['worst_case_profit']:.2f}")
        print(f"利润标准差: {result['profit_std']:.2f}")
        print(f"决策置信度: {result['decision_confidence']*100:.1f}%")
        print(f"模拟次数: {result['simulation_count']}")
        print(f"求解时间: {result['solution_time']*1000:.2f}ms")
        
        # 创建可视化
        print("\n正在生成可视化图表...")
        paths = create_robust_dashboard(result, base_params)
        print(f"图表已保存:")
        for i, path in enumerate(paths, 1):
            print(f"{i}. {path}")
        
        # 测试多工序鲁棒优化
        print("\n测试多工序鲁棒优化...")
        graph = create_example_network()
        result = robust_optimize_multistage(graph, uncertainty_params)
        
        print(f"\n各节点鲁棒决策:")
        for node, decision in result['robust_decisions'].items():
            print(f"\n节点 {node}:")
            print(f"  检测: {'是' if decision['test'] else '否'}")
            print(f"  返修: {'是' if decision['repair'] else '否'}")
            print(f"  决策置信度: {decision['decision_confidence']*100:.1f}%")
            
        print(f"\n期望总成本: {result['expected_cost']:.2f}")
        print(f"最差情况成本: {result['worst_case_cost']:.2f}")
        print(f"成本标准差: {result['cost_std']:.2f}")
        print(f"模拟次数: {result['simulation_count']}")
        print(f"求解时间: {result['solution_time']*1000:.2f}ms")
        
        # 创建可视化
        print("\n正在生成可视化图表...")
        paths = create_robust_dashboard(result, graph=graph)
        print(f"图表已保存:")
        for i, path in enumerate(paths, 1):
            print(f"{i}. {path}")
        
    except Exception as e:
        logger.error(f"鲁棒优化分析失败: {str(e)}")

def generate_report():
    """生成完整的论文报告"""
    logger.info("开始生成论文报告...")
    
    try:
        # 收集抽样检验结果
        sampling_results = []
        # 基准情况
        n, c, alpha, beta = optimal_sampling(p0=0.1, alpha=0.05,
                                          beta=0.1, p1=0.15)
        sampling_results.append({
            'n': n,
            'c': c,
            'alpha': alpha,
            'beta': beta
        })
        # 更严格的情况
        n, c, alpha, beta = optimal_sampling(p0=0.1, alpha=0.01,
                                          beta=0.05, p1=0.15)
        sampling_results.append({
            'n': n,
            'c': c,
            'alpha': alpha,
            'beta': beta
        })
        
        # 收集生产决策结果
        production_results = []
        # 基准情况
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
        result = optimize_production(params)
        production_results.append(result)
        
        # 高不合格率情况
        params.defect_rate1 = 0.15
        params.defect_rate2 = 0.15
        result = optimize_production(params)
        production_results.append(result)
        
        # 高检测成本情况
        params.defect_rate1 = 0.1
        params.defect_rate2 = 0.1
        params.test_cost1 = 4
        params.test_cost2 = 6
        result = optimize_production(params)
        production_results.append(result)
        
        # 收集多工序优化结果
        graph = create_example_network()
        multistage_result = optimize_multistage(graph)
        
        # 收集鲁棒优化结果（使用较小的样本量）
        uncertainty_params = UncertaintyParams(
            n_samples=50,
            n_simulations=50,
            confidence_level=0.95
        )
        
        robust_results = {
            'production': robust_optimize_production(params, uncertainty_params),
            'multistage': robust_optimize_multistage(graph, uncertainty_params)
        }
        
        # 生成论文
        paper = generate_paper(
            sampling_results=sampling_results,
            production_results=production_results,
            multistage_result=multistage_result,
            robust_results=robust_results
        )
        
        # 保存论文
        filename = save_paper(paper)
        logger.info(f"论文已保存到: {filename}")
        
    except Exception as e:
        logger.error(f"生成论文失败: {str(e)}")

def main():
    """主函数"""
    print("=== 2024年数学建模B题求解系统 ===\n")
    
    # 问题1：抽样检验方案
    print("【问题1】抽样检验方案")
    analyze_sampling()
    
    # 问题2：生产决策优化
    print("\n【问题2】生产决策优化")
    analyze_production()
    
    # 问题3：多工序扩展
    print("\n【问题3】多工序扩展")
    analyze_multistage()
    
    # 问题4：鲁棒优化
    print("\n【问题4】鲁棒优化")
    analyze_robust()
    
    # 生成论文报告
    print("\n【生成论文】")
    generate_report()

if __name__ == "__main__":
    main() 