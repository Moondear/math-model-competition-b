import pytest
import numpy as np
from src.sampling import SamplingPlanOptimizer, optimal_sampling

def test_parameter_validation():
    """测试参数验证"""
    with pytest.raises(ValueError):
        SamplingPlanOptimizer(p0=-0.1)  # 非法p0
    with pytest.raises(ValueError):
        SamplingPlanOptimizer(p0=0.2, p1=0.1)  # p0 > p1
    with pytest.raises(ValueError):
        SamplingPlanOptimizer(alpha=1.5)  # 非法alpha
    with pytest.raises(ValueError):
        SamplingPlanOptimizer(beta=-0.1)  # 非法beta

def test_error_probability_calculation():
    """测试错误概率计算"""
    optimizer = SamplingPlanOptimizer(p0=0.1, p1=0.15)
    alpha, beta = optimizer._calculate_error_probs(n=100, c=15)
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1

def test_optimal_sampling_result():
    """测试最优抽样方案"""
    n, c, alpha, beta = optimal_sampling(
        p0=0.1, p1=0.15, alpha=0.05, beta=0.1
    )
    assert isinstance(n, int) and n > 0
    assert isinstance(c, int) and 0 <= c <= n
    assert 0 <= alpha <= 0.05  # 满足α要求
    assert 0 <= beta <= 0.1   # 满足β要求

def test_large_scale_performance():
    """测试大规模计算性能"""
    start_time = np.datetime64('now')
    optimal_sampling(max_n=1000)
    end_time = np.datetime64('now')
    duration = (end_time - start_time).astype('timedelta64[ms]').astype(int)
    assert duration < 60000  # 确保在60秒内完成

def test_parallel_optimization():
    """测试并行优化"""
    optimizer = SamplingPlanOptimizer()
    result1 = optimizer.optimize(max_n=100, n_workers=1)
    result2 = optimizer.optimize(max_n=100, n_workers=4)
    assert result1['n'] == result2['n']  # 结果一致性
    assert result1['c'] == result2['c']

def test_stress_test():
    """压力测试"""
    test_cases = [
        (0.05, 0.10, 0.05, 0.10),
        (0.10, 0.20, 0.01, 0.01),
        (0.01, 0.05, 0.10, 0.10)
    ]
    for p0, p1, alpha, beta in test_cases:
        n, c, _, _ = optimal_sampling(p0, alpha, beta, p1)
        assert n > 0 and c >= 0

def test_edge_cases():
    """测试边界情况"""
    # 极小的p0和p1差异
    n, c, alpha, beta = optimal_sampling(
        p0=0.1, p1=0.101, alpha=0.05, beta=0.1
    )
    assert n > 0  # 应该需要更大的样本量

    # 极小的错误概率
    n, c, alpha, beta = optimal_sampling(
        p0=0.1, p1=0.15, alpha=0.01, beta=0.01
    )
    assert n > 0  # 应该需要更大的样本量

if __name__ == "__main__":
    pytest.main(["-v"]) 