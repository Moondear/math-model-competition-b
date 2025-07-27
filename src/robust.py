"""
鲁棒优化模块
"""
import numpy as np
from scipy.stats import beta
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import logging
from src.production import ProductionParams, optimize_production
from src.multistage import NodeParams, optimize_multistage
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UncertaintyParams:
    """不确定性参数"""
    n_samples: int = 50     # 减少抽样数量
    n_simulations: int = 50  # 减少模拟次数
    confidence_level: float = 0.95  # 置信水平
    
    def validate(self):
        """验证参数有效性"""
        if self.n_samples <= 0 or self.n_simulations <= 0:
            raise ValueError("样本量和模拟次数必须为正数")
        if not 0 < self.confidence_level < 1:
            raise ValueError("置信水平必须在(0,1)范围内")

class RobustOptimizer:
    """鲁棒优化器"""
    
    def __init__(self, uncertainty_params: UncertaintyParams):
        """初始化优化器
        
        Args:
            uncertainty_params: 不确定性参数
        """
        self.params = uncertainty_params
        self.params.validate()
        
    def _generate_beta_samples(self, p_hat: float, n: int) -> np.ndarray:
        """生成Beta分布样本
        
        Args:
            p_hat: 观测的不合格率
            n: 样本量
            
        Returns:
            np.ndarray: Beta分布样本
        """
        # 增加不确定性：使用较小的样本量来增加方差
        effective_n = min(n, 20)  # 限制有效样本量以增加不确定性
        k = int(p_hat * effective_n)  # 观测到的不合格品数量
        alpha = k + 1
        beta_param = effective_n - k + 1
        return beta.rvs(alpha, beta_param, size=self.params.n_samples)
        
    def _evaluate_scenario(self, scenario_params: Dict) -> Dict:
        """评估单个场景
        
        Args:
            scenario_params: 场景参数
            
        Returns:
            Dict: 评估结果
        """
        try:
            result = optimize_production(ProductionParams(**scenario_params))
            return {
                'profit': result['expected_profit'],
                'decisions': {
                    'test_part1': result['test_part1'],
                    'test_part2': result['test_part2'],
                    'test_final': result['test_final'],
                    'repair': result['repair']
                }
            }
        except Exception as e:
            logger.error(f"场景评估失败: {str(e)}")
            return None
            
    def optimize_production(self, base_params: ProductionParams) -> Dict:
        """优化生产决策
        
        Args:
            base_params: 基准参数
            
        Returns:
            Dict: 优化结果
        """
        try:
            start_time = time.time()
            
            # 生成不确定参数场景
            results = []
            for _ in range(self.params.n_simulations):
                # 扰动不合格率
                p1_samples = self._generate_beta_samples(base_params.defect_rate1, 
                                                       self.params.n_samples)
                p2_samples = self._generate_beta_samples(base_params.defect_rate2, 
                                                       self.params.n_samples)
                
                # 取随机样本而不是置信区间上界
                p1 = np.random.choice(p1_samples)
                p2 = np.random.choice(p2_samples)
                
                # 同时扰动其他参数
                scenario = base_params.__dict__.copy()
                scenario['defect_rate1'] = p1
                scenario['defect_rate2'] = p2
                # 添加成本参数的随机扰动（±10%）
                for param in ['test_cost1', 'test_cost2', 'assembly_cost', 
                            'test_cost_final', 'repair_cost']:
                    base_value = scenario[param]
                    scenario[param] = base_value * np.random.uniform(0.9, 1.1)
                
                # 评估场景
                result = self._evaluate_scenario(scenario)
                if result is not None:
                    results.append(result)
            
            if not results:
                raise ValueError("所有场景评估均失败")
                
            # 统计分析
            profits = [r['profit'] for r in results]
            decisions = [r['decisions'] for r in results]
            
            # 找出最常见的决策组合
            decision_strs = [''.join(str(int(v)) for v in d.values()) 
                           for d in decisions]
            unique_decisions, counts = np.unique(decision_strs, 
                                               return_counts=True)
            robust_decision_str = unique_decisions[counts.argmax()]
            
            # 转回决策字典
            robust_decision = {
                'test_part1': bool(int(robust_decision_str[0])),
                'test_part2': bool(int(robust_decision_str[1])),
                'test_final': bool(int(robust_decision_str[2])),
                'repair': bool(int(robust_decision_str[3]))
            }
            
            # 计算鲁棒性指标
            solution = {
                'robust_decision': robust_decision,
                'expected_profit': np.mean(profits),
                'worst_case_profit': np.min(profits),
                'profit_std': np.std(profits),
                'decision_confidence': counts.max() / len(decisions),
                'simulation_count': len(results),
                'solution_time': time.time() - start_time,
                'simulation_results': results  # 添加模拟结果
            }
            
            return solution
            
        except Exception as e:
            logger.error(f"鲁棒优化失败: {str(e)}")
            raise
            
    def optimize_multistage(self, graph: nx.DiGraph) -> Dict:
        """优化多工序系统
        
        Args:
            graph: 生产网络
            
        Returns:
            Dict: 优化结果
        """
        try:
            start_time = time.time()
            
            # 生成场景
            results = []
            for _ in range(self.params.n_simulations):
                scenario_graph = graph.copy()
                
                # 扰动每个节点的不合格率
                for node in graph.nodes:
                    params = graph.nodes[node]['params']
                    samples = self._generate_beta_samples(params.defect_rate,
                                                        self.params.n_samples)
                    p = np.random.choice(samples)  # 使用随机采样
                    
                    # 更新节点参数
                    new_params = NodeParams(
                        defect_rate=p,
                        process_cost=params.process_cost * np.random.uniform(0.9, 1.1),
                        test_cost=params.test_cost * np.random.uniform(0.9, 1.1),
                        repair_cost=params.repair_cost * np.random.uniform(0.9, 1.1)
                    )
                    scenario_graph.nodes[node]['params'] = new_params
                    
                # 评估场景
                try:
                    result = optimize_multistage(scenario_graph)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"场景评估失败: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("所有场景评估均失败")
                
            # 统计分析
            costs = [r['total_cost'] for r in results]
            decisions = [r['decisions'] for r in results]
            
            # 对每个节点找出最常见的决策组合
            robust_decisions = {}
            for node in graph.nodes:
                node_decisions = []
                for d in decisions:
                    decision_str = ''.join(str(int(v)) 
                                         for v in [d[node]['test'], 
                                                 d[node]['repair']])
                    node_decisions.append(decision_str)
                    
                unique_decisions, counts = np.unique(node_decisions, 
                                                   return_counts=True)
                robust_decision_str = unique_decisions[counts.argmax()]
                
                robust_decisions[node] = {
                    'test': bool(int(robust_decision_str[0])),
                    'repair': bool(int(robust_decision_str[1])),
                    'decision_confidence': counts.max() / len(decisions)
                }
            
            # 计算鲁棒性指标
            solution = {
                'robust_decisions': robust_decisions,
                'expected_cost': np.mean(costs),
                'worst_case_cost': np.max(costs),
                'cost_std': np.std(costs),
                'simulation_count': len(results),
                'solution_time': time.time() - start_time,
                'simulation_results': results  # 添加模拟结果
            }
            
            return solution
            
        except Exception as e:
            logger.error(f"多工序鲁棒优化失败: {str(e)}")
            raise

def robust_optimize_production(base_params: ProductionParams,
                             uncertainty_params: Optional[UncertaintyParams] = None
                             ) -> Dict:
    """鲁棒生产决策优化
    
    Args:
        base_params: 基准参数
        uncertainty_params: 不确定性参数
        
    Returns:
        Dict: 优化结果
    """
    try:
        if uncertainty_params is None:
            uncertainty_params = UncertaintyParams()
            
        optimizer = RobustOptimizer(uncertainty_params)
        return optimizer.optimize_production(base_params)
    except Exception as e:
        logger.error(f"鲁棒生产优化失败: {str(e)}")
        raise

def robust_optimize_multistage(graph: nx.DiGraph,
                             uncertainty_params: Optional[UncertaintyParams] = None
                             ) -> Dict:
    """鲁棒多工序优化
    
    Args:
        graph: 生产网络
        uncertainty_params: 不确定性参数
        
    Returns:
        Dict: 优化结果
    """
    try:
        if uncertainty_params is None:
            uncertainty_params = UncertaintyParams()
            
        optimizer = RobustOptimizer(uncertainty_params)
        return optimizer.optimize_multistage(graph)
    except Exception as e:
        logger.error(f"鲁棒多工序优化失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试鲁棒优化器
    try:
        # 测试生产决策优化
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
        
        uncertainty_params = UncertaintyParams(
            n_samples=100,
            n_simulations=100,
            confidence_level=0.95
        )
        
        print("\n测试生产决策鲁棒优化...")
        result = robust_optimize_production(base_params, uncertainty_params)
        
        print(f"鲁棒决策:")
        for key, value in result['robust_decision'].items():
            print(f"  {key}: {'是' if value else '否'}")
        print(f"期望利润: {result['expected_profit']:.2f}")
        print(f"最差情况利润: {result['worst_case_profit']:.2f}")
        print(f"利润标准差: {result['profit_std']:.2f}")
        print(f"决策置信度: {result['decision_confidence']:.2%}")
        print(f"模拟次数: {result['simulation_count']}")
        print(f"求解时间: {result['solution_time']*1000:.2f}ms")
        
        # 测试多工序优化
        from src.multistage import create_example_network
        
        print("\n测试多工序鲁棒优化...")
        graph = create_example_network()
        result = robust_optimize_multistage(graph, uncertainty_params)
        
        print(f"\n各节点鲁棒决策:")
        for node, decision in result['robust_decisions'].items():
            print(f"\n节点 {node}:")
            print(f"  检测: {'是' if decision['test'] else '否'}")
            print(f"  返修: {'是' if decision['repair'] else '否'}")
            print(f"  决策置信度: {decision['decision_confidence']:.2%}")
            
        print(f"\n期望总成本: {result['expected_cost']:.2f}")
        print(f"最差情况成本: {result['worst_case_cost']:.2f}")
        print(f"成本标准差: {result['cost_std']:.2f}")
        print(f"模拟次数: {result['simulation_count']}")
        print(f"求解时间: {result['solution_time']*1000:.2f}ms")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}") 