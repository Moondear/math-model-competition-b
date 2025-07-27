import networkx as nx
from ortools.linear_solver import pywraplp
import numpy as np
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessNode:
    """工序节点数据类"""
    node_id: str
    defect_rate: float
    test_cost: float
    process_cost: float
    repair_cost: Optional[float] = None
    is_final: bool = False
    market_price: Optional[float] = None
    return_loss: Optional[float] = None

class MultiStageOptimizer:
    """多工序生产优化器
    
    使用NetworkX构建生产网络，OR-Tools求解优化问题
    """
    
    def __init__(self):
        """初始化优化器"""
        self.graph = nx.DiGraph()
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self.solver:
            raise RuntimeError("SCIP求解器初始化失败")
            
        self.solver.SetNumThreads(8)
        
    def add_node(self, node: ProcessNode) -> None:
        """添加工序节点
        
        Args:
            node: 工序节点数据
        """
        self.graph.add_node(
            node.node_id,
            defect_rate=node.defect_rate,
            test_cost=node.test_cost,
            process_cost=node.process_cost,
            repair_cost=node.repair_cost,
            is_final=node.is_final,
            market_price=node.market_price,
            return_loss=node.return_loss
        )
        
    def add_edge(self, from_id: str, to_id: str) -> None:
        """添加工序流向关系
        
        Args:
            from_id: 起始节点ID
            to_id: 目标节点ID
        """
        self.graph.add_edge(from_id, to_id)
        
    def _create_variables(self) -> None:
        """创建决策变量"""
        self.vars = {}
        for node in self.graph.nodes:
            # 检测决策
            self.vars[f'test_{node}'] = self.solver.BoolVar(
                f'test_{node}'
            )
            
            # 拆解决策（非原料节点）
            if self.graph.in_degree(node) > 0:
                self.vars[f'repair_{node}'] = self.solver.BoolVar(
                    f'repair_{node}'
                )
                
            # 合格概率
            self.vars[f'p_ok_{node}'] = self.solver.NumVar(
                0, 1, f'p_ok_{node}'
            )
            
    def _add_probability_constraints(self) -> None:
        """添加概率传播约束"""
        for node in nx.topological_sort(self.graph):
            # 获取前序节点
            predecessors = list(self.graph.predecessors(node))
            
            if not predecessors:  # 原料节点
                p = self.graph.nodes[node]['defect_rate']
                x = self.vars[f'test_{node}']
                self.solver.Add(
                    self.vars[f'p_ok_{node}'] <= 1 - p * (1 - x)
                )
            else:  # 中间/成品节点
                # 前序节点合格概率的乘积
                pred_prob = 1.0
                for pred in predecessors:
                    pred_prob *= self.vars[f'p_ok_{pred}']
                    
                # 本节点的合格概率
                p = self.graph.nodes[node]['defect_rate']
                x = self.vars[f'test_{node}']
                self.solver.Add(
                    self.vars[f'p_ok_{node}'] <= pred_prob * (1 - p * (1 - x))
                )
                
    def _add_repair_constraints(self) -> None:
        """添加拆解约束"""
        for node in self.graph.nodes:
            if self.graph.in_degree(node) > 0:
                # 只有检测才能拆解
                self.solver.Add(
                    self.vars[f'repair_{node}'] <= 
                    self.vars[f'test_{node}']
                )
                
    def _build_objective(self) -> None:
        """构建目标函数"""
        obj_expr = self.solver.NumVar(
            -self.solver.infinity(),
            self.solver.infinity(),
            'objective'
        )
        
        # 收入（只考虑最终产品）
        revenue = 0
        for node in self.graph.nodes:
            if self.graph.nodes[node]['is_final']:
                p_ok = self.vars[f'p_ok_{node}']
                price = self.graph.nodes[node]['market_price']
                loss = self.graph.nodes[node]['return_loss']
                revenue += price * p_ok - loss * (1 - p_ok)
                
        # 成本
        costs = 0
        for node in self.graph.nodes:
            # 检测成本
            costs += (
                self.vars[f'test_{node}'] * 
                self.graph.nodes[node]['test_cost']
            )
            
            # 加工成本
            costs += self.graph.nodes[node]['process_cost']
            
            # 拆解成本
            if self.graph.in_degree(node) > 0:
                costs += (
                    self.vars[f'repair_{node}'] * 
                    self.graph.nodes[node]['repair_cost']
                )
                
        # 最大化利润
        self.solver.Add(obj_expr == revenue - costs)
        self.solver.Maximize(obj_expr)
        
    def solve(self, timeout_sec: int = 60) -> Dict:
        """求解优化问题
        
        Args:
            timeout_sec: 求解超时时间(秒)
            
        Returns:
            Dict: 优化结果
        """
        start_time = time.time()
        
        # 创建变量
        self._create_variables()
        
        # 添加约束
        self._add_probability_constraints()
        self._add_repair_constraints()
        
        # 构建目标函数
        self._build_objective()
        
        # 设置超时
        self.solver.SetTimeLimit(timeout_sec * 1000)
        
        # 求解
        status = self.solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            logger.warning("未找到最优解，使用启发式方法")
            return self._fallback_heuristic()
            
        # 提取结果
        solution = {
            'decisions': {},
            'probabilities': {},
            'objective_value': self.solver.Objective().Value(),
            'computation_time': time.time() - start_time,
            'status': 'optimal'
        }
        
        for node in self.graph.nodes:
            # 检测决策
            solution['decisions'][f'test_{node}'] = bool(
                self.vars[f'test_{node}'].solution_value()
            )
            
            # 拆解决策
            if self.graph.in_degree(node) > 0:
                solution['decisions'][f'repair_{node}'] = bool(
                    self.vars[f'repair_{node}'].solution_value()
                )
                
            # 合格概率
            solution['probabilities'][node] = float(
                self.vars[f'p_ok_{node}'].solution_value()
            )
            
        return solution
        
    def _fallback_heuristic(self) -> Dict:
        """启发式求解方法
        
        采用贪心策略：
        1. 按拓扑序遍历节点
        2. 对每个节点，如果检测成本小于预期损失，则进行检测
        3. 对每个节点，如果拆解成本小于预期损失，则进行拆解
        """
        solution = {
            'decisions': {},
            'probabilities': {},
            'status': 'heuristic'
        }
        
        # 按拓扑序处理节点
        for node in nx.topological_sort(self.graph):
            data = self.graph.nodes[node]
            
            # 计算前序节点的合格概率
            pred_prob = 1.0
            for pred in self.graph.predecessors(node):
                pred_prob *= solution['probabilities'][pred]
                
            # 计算本节点的合格概率
            p_defect = data['defect_rate']
            
            # 检测决策
            expected_loss = p_defect * (
                data['market_price'] if data['is_final']
                else sum(self.graph.nodes[succ]['process_cost']
                        for succ in self.graph.successors(node))
            )
            should_test = data['test_cost'] < expected_loss
            solution['decisions'][f'test_{node}'] = should_test
            
            # 拆解决策
            if self.graph.in_degree(node) > 0:
                should_repair = (
                    should_test and
                    data['repair_cost'] < expected_loss
                )
                solution['decisions'][f'repair_{node}'] = should_repair
                
            # 更新合格概率
            p_ok = pred_prob * (
                1 - p_defect if should_test
                else 1 - p_defect * 0.5  # 简化假设
            )
            solution['probabilities'][node] = p_ok
            
        return solution

def create_production_network() -> MultiStageOptimizer:
    """创建示例生产网络
    
    Returns:
        MultiStageOptimizer: 配置好的优化器
    """
    optimizer = MultiStageOptimizer()
    
    # 添加节点
    nodes = [
        ProcessNode("P1", 0.1, 2, 1),
        ProcessNode("P2", 0.1, 8, 1),
        ProcessNode("P3", 0.1, 12, 2),
        ProcessNode("P4", 0.1, 2, 1),
        ProcessNode("P5", 0.1, 8, 1),
        ProcessNode("P6", 0.1, 12, 2),
        ProcessNode("P7", 0.1, 8, 1),
        ProcessNode("P8", 0.1, 12, 2),
        ProcessNode("A1", 0.1, 8, 4, repair_cost=6),
        ProcessNode("A2", 0.1, 8, 4, repair_cost=6),
        ProcessNode("A3", 0.1, 8, 4, repair_cost=6),
        ProcessNode("F", 0.1, 8, 6, repair_cost=10, 
                   is_final=True, market_price=200, return_loss=40)
    ]
    
    for node in nodes:
        optimizer.add_node(node)
        
    # 添加边
    edges = [
        ("P1", "A1"), ("P2", "A1"), ("P3", "A1"),
        ("P4", "A2"), ("P5", "A2"), ("P6", "A2"),
        ("P7", "A3"), ("P8", "A3"),
        ("A1", "F"), ("A2", "F"), ("A3", "F")
    ]
    
    for from_id, to_id in edges:
        optimizer.add_edge(from_id, to_id)
        
    return optimizer

if __name__ == "__main__":
    # 运行单元测试
    import pytest
    pytest.main(["-v", "tests/test_multi_stage.py"]) 