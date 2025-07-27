import pytest
import numpy as np
from src.production import ProductionParams, ProductionOptimizer, optimize_production

@pytest.fixture
def default_params():
    """默认生产参数"""
    return ProductionParams(
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

def test_initialization(default_params):
    """测试优化器初始化"""
    optimizer = ProductionOptimizer(default_params)
    assert optimizer.solver is not None
    assert optimizer.x1 is not None
    assert optimizer.x2 is not None
    assert optimizer.y is not None
    assert optimizer.z is not None

def test_constraint_satisfaction(default_params):
    """测试约束满足情况"""
    optimizer = ProductionOptimizer(default_params)
    solution = optimizer.solve()
    
    # 检查拆解逻辑约束
    assert not (solution['repair'] and not solution['test_final'])
    
    # 检查决策变量取值范围
    assert isinstance(solution['test_part1'], bool)
    assert isinstance(solution['test_part2'], bool)
    assert isinstance(solution['test_final'], bool)
    assert isinstance(solution['repair'], bool)

def test_profit_calculation(default_params):
    """测试利润计算"""
    optimizer = ProductionOptimizer(default_params)
    solution = optimizer.solve()
    
    # 利润应该是有限值
    assert np.isfinite(solution['expected_profit'])
    
    # 利润应该大于最小成本
    min_cost = default_params.assembly_cost
    assert solution['expected_profit'] > -min_cost

def test_computation_time(default_params):
    """测试计算时间"""
    optimizer = ProductionOptimizer(default_params)
    solution = optimizer.solve(timeout_sec=1)
    assert 'computation_time' in solution
    assert solution['computation_time'] < 2  # 包含一些额外开销

def test_parameter_sensitivity(default_params):
    """测试参数敏感性"""
    # 增加市场价格
    high_price_params = default_params
    high_price_params.market_price *= 2
    solution1 = optimize_production(high_price_params)
    
    # 减少市场价格
    low_price_params = default_params
    low_price_params.market_price *= 0.5
    solution2 = optimize_production(low_price_params)
    
    # 高价格应该带来更高利润
    assert solution1['expected_profit'] > solution2['expected_profit']

def test_extreme_cases(default_params):
    """测试极端情况"""
    # 极高的不合格率
    extreme_params = default_params
    extreme_params.defect_rate1 = 0.9
    extreme_params.defect_rate2 = 0.9
    solution = optimize_production(extreme_params)
    assert solution['status'] in ['optimal', 'heuristic']

def test_fallback_mechanism(default_params):
    """测试回退机制"""
    # 设置一个极短的超时时间触发回退
    optimizer = ProductionOptimizer(default_params)
    solution = optimizer.solve(timeout_sec=0.001)
    assert solution['status'] == 'heuristic'

def test_solution_stability(default_params):
    """测试解的稳定性"""
    # 多次求解结果应该一致
    optimizer = ProductionOptimizer(default_params)
    solution1 = optimizer.solve()
    solution2 = optimizer.solve()
    
    assert solution1['test_part1'] == solution2['test_part1']
    assert solution1['test_part2'] == solution2['test_part2']
    assert solution1['test_final'] == solution2['test_final']
    assert solution1['repair'] == solution2['repair']

def test_large_scale_performance():
    """大规模性能测试"""
    # 生成多组随机参数
    n_tests = 100
    for _ in range(n_tests):
        params = ProductionParams(
            defect_rate1=np.random.uniform(0.05, 0.2),
            defect_rate2=np.random.uniform(0.05, 0.2),
            test_cost1=np.random.uniform(1, 5),
            test_cost2=np.random.uniform(1, 5),
            assembly_cost=np.random.uniform(4, 8),
            test_cost_final=np.random.uniform(2, 4),
            repair_cost=np.random.uniform(4, 6),
            market_price=np.random.uniform(40, 60),
            return_loss=np.random.uniform(5, 10)
        )
        solution = optimize_production(params)
        assert solution['status'] in ['optimal', 'heuristic']

if __name__ == "__main__":
    pytest.main(["-v"]) 