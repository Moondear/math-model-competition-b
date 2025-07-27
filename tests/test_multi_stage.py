import pytest
import networkx as nx
from src.multi_stage import (
    ProcessNode, MultiStageOptimizer, create_production_network
)

@pytest.fixture
def simple_network():
    """创建简单的测试网络"""
    optimizer = MultiStageOptimizer()
    
    # 添加节点
    nodes = [
        ProcessNode("P1", 0.1, 2, 1),
        ProcessNode("P2", 0.1, 2, 1),
        ProcessNode("A1", 0.1, 3, 2, repair_cost=4),
        ProcessNode("F", 0.1, 3, 2, repair_cost=5,
                   is_final=True, market_price=20, return_loss=4)
    ]
    
    for node in nodes:
        optimizer.add_node(node)
        
    # 添加边
    edges = [("P1", "A1"), ("P2", "A1"), ("A1", "F")]
    for from_id, to_id in edges:
        optimizer.add_edge(from_id, to_id)
        
    return optimizer

def test_network_creation(simple_network):
    """测试网络创建"""
    assert len(simple_network.graph.nodes) == 4
    assert len(simple_network.graph.edges) == 3
    
    # 检查节点属性
    assert simple_network.graph.nodes["F"]["is_final"]
    assert simple_network.graph.nodes["A1"]["repair_cost"] == 4
    
    # 检查网络结构
    assert nx.is_directed_acyclic_graph(simple_network.graph)

def test_variable_creation(simple_network):
    """测试变量创建"""
    simple_network._create_variables()
    
    # 检查变量数量
    n_nodes = len(simple_network.graph.nodes)
    n_assembly = sum(1 for n in simple_network.graph.nodes
                    if simple_network.graph.in_degree(n) > 0)
    
    expected_vars = n_nodes * 2 + n_assembly  # test + p_ok + repair
    assert len(simple_network.vars) == expected_vars
    
    # 检查变量类型
    assert all(var.Integer() for name, var in simple_network.vars.items()
              if name.startswith('test_') or name.startswith('repair_'))
    assert all(not var.Integer() for name, var in simple_network.vars.items()
              if name.startswith('p_ok_'))

def test_constraint_creation(simple_network):
    """测试约束创建"""
    simple_network._create_variables()
    
    # 添加概率约束
    simple_network._add_probability_constraints()
    constraints1 = simple_network.solver.NumConstraints()
    
    # 添加拆解约束
    simple_network._add_repair_constraints()
    constraints2 = simple_network.solver.NumConstraints()
    
    assert constraints2 > constraints1  # 确保添加了新约束

def test_optimization_result(simple_network):
    """测试优化结果"""
    solution = simple_network.solve()
    
    # 检查结果格式
    assert 'decisions' in solution
    assert 'probabilities' in solution
    assert 'status' in solution
    
    # 检查决策变量取值
    for node in simple_network.graph.nodes:
        assert f'test_{node}' in solution['decisions']
        if simple_network.graph.in_degree(node) > 0:
            assert f'repair_{node}' in solution['decisions']
            
    # 检查概率取值
    for node in simple_network.graph.nodes:
        assert node in solution['probabilities']
        assert 0 <= solution['probabilities'][node] <= 1

def test_fallback_mechanism(simple_network):
    """测试回退机制"""
    # 设置极短的超时时间触发回退
    solution = simple_network.solve(timeout_sec=0.001)
    assert solution['status'] == 'heuristic'
    
    # 检查回退解的合理性
    for node in simple_network.graph.nodes:
        if solution['decisions'].get(f'repair_{node}', False):
            assert solution['decisions'][f'test_{node}']

def test_full_network():
    """测试完整网络"""
    optimizer = create_production_network()
    solution = optimizer.solve()
    
    # 检查网络规模
    assert len(optimizer.graph.nodes) == 12
    assert len(optimizer.graph.edges) == 11
    
    # 检查解的完整性
    assert len(solution['decisions']) == len(optimizer.graph.nodes) * 2 - 8
    assert len(solution['probabilities']) == len(optimizer.graph.nodes)

def test_stress_test():
    """压力测试"""
    optimizer = create_production_network()
    
    # 多次求解
    n_tests = 10
    solutions = []
    for _ in range(n_tests):
        solution = optimizer.solve(timeout_sec=1)
        solutions.append(solution)
        
    # 检查解的稳定性
    first = solutions[0]
    for other in solutions[1:]:
        assert first['decisions'] == other['decisions']
        for node in first['probabilities']:
            assert abs(first['probabilities'][node] - 
                      other['probabilities'][node]) < 1e-6

def test_edge_cases():
    """测试边界情况"""
    optimizer = MultiStageOptimizer()
    
    # 测试单节点
    optimizer.add_node(ProcessNode(
        "P1", 0.1, 2, 1, is_final=True,
        market_price=10, return_loss=2
    ))
    solution = optimizer.solve()
    assert solution['status'] in ['optimal', 'heuristic']
    
    # 测试线性链
    optimizer = MultiStageOptimizer()
    nodes = [
        ProcessNode("P1", 0.1, 2, 1),
        ProcessNode("P2", 0.1, 2, 1, repair_cost=3),
        ProcessNode("P3", 0.1, 2, 1, repair_cost=3,
                   is_final=True, market_price=10, return_loss=2)
    ]
    for node in nodes:
        optimizer.add_node(node)
    optimizer.add_edge("P1", "P2")
    optimizer.add_edge("P2", "P3")
    solution = optimizer.solve()
    assert solution['status'] in ['optimal', 'heuristic']

if __name__ == "__main__":
    pytest.main(["-v"]) 