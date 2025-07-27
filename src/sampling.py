"""
抽样检验方案模块
"""
import numpy as np
from scipy.stats import binom
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimal_sampling(p0: float = 0.1, alpha: float = 0.05,
                    beta: float = 0.1, p1: float = 0.15,
                    max_n: int = 2000) -> tuple[int, int, float, float]:
    """鲁棒性抽样方案设计
    
    Args:
        p0: 原假设下的不合格率
        alpha: 第一类错误概率
        beta: 第二类错误概率
        p1: 备择假设下的不合格率
        max_n: 最大样本量
        
    Returns:
        tuple[int, int, float, float]: (n, c, 实际α, 实际β)
    """
    # 参数验证
    if not (0 < p0 < p1 < 1):
        raise ValueError("必须满足: 0 < p0 < p1 < 1")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("必须满足: 0 < alpha,beta < 1")
        
    best_result = None
    min_n = float('inf')
    
    # 使用二分查找加速
    n_low = 10
    n_high = max_n
    
    while n_low <= n_high:
        n = (n_low + n_high) // 2
        found_valid_c = False
        
        # 对当前n寻找最小的可行c
        for c in range(n + 1):
            # 计算实际的α和β
            actual_alpha = 1 - binom.cdf(c, n, p0)
            actual_beta = binom.cdf(c, n, p1)
            
            # 检查是否满足要求
            if actual_alpha <= alpha and actual_beta <= beta:
                found_valid_c = True
                if n < min_n:
                    min_n = n
                    best_result = (n, c, actual_alpha, actual_beta)
                break
        
        # 根据结果调整搜索范围
        if found_valid_c:
            n_high = n - 1  # 尝试更小的n
        else:
            n_low = n + 1   # 需要更大的n
            
    if best_result is None:
        raise ValueError(f"在max_n={max_n}范围内未找到解")
        
    return best_result

def run_stress_test(n_iterations: int = 1000) -> None:
    """运行压力测试
    
    Args:
        n_iterations: 迭代次数
    """
    logger.info(f"开始压力测试 ({n_iterations}次迭代)...")
    start_time = time.time()
    
    success_count = 0
    failure_count = 0
    
    for i in range(n_iterations):
        if i % 100 == 0:
            logger.info(f"测试进度: {i}/{n_iterations}")
            
        try:
            # 随机生成合理范围内的参数
            p0 = np.random.uniform(0.05, 0.15)
            p1 = p0 + np.random.uniform(0.03, 0.05)  # 确保p1 > p0
            alpha = np.random.uniform(0.03, 0.1)
            beta = np.random.uniform(0.05, 0.15)
            
            # 运行优化
            n, c, alpha_actual, beta_actual = optimal_sampling(
                p0=p0, p1=p1, alpha=alpha, beta=beta
            )
            
            # 验证结果
            if (alpha_actual <= alpha and beta_actual <= beta and
                n > 0 and 0 <= c <= n):
                success_count += 1
            else:
                failure_count += 1
                logger.warning(f"测试{i}: 结果验证失败")
                
        except Exception as e:
            failure_count += 1
            logger.error(f"测试{i}失败: {str(e)}")
            
    # 输出测试报告
    total_time = time.time() - start_time
    logger.info("\n压力测试报告:")
    logger.info(f"总测试数: {n_iterations}")
    logger.info(f"成功数: {success_count}")
    logger.info(f"失败数: {failure_count}")
    logger.info(f"成功率: {success_count/n_iterations*100:.2f}%")
    logger.info(f"平均耗时: {total_time/n_iterations*1000:.2f}ms/次")
    logger.info(f"总耗时: {total_time:.2f}秒") 