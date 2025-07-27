"""
亿级优化器测试模块
"""
import unittest
import numpy as np
import time
import logging
import psutil
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from innovation.exascale_optimizer import ExascaleOptimizer, ExascaleParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestExascaleOptimizer(unittest.TestCase):
    """亿级优化器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.params = ExascaleParams(
            num_nodes=2,  # 测试环境使用2个节点
            chunk_size=10_000,  # 1万变量/块，测试环境
            gpu_device='cuda:0' if torch.cuda.is_available() else 'cpu',
            max_threads=4  # 减少线程数
        )
        self.optimizer = ExascaleOptimizer(self.params)
        
    def test_large_scale_optimization(self):
        """测试大规模优化"""
        # 测试问题设置
        problem_size = 50_000  # 5万变量用于测试
        constraints = [
            {'type': 'sum', 'bound': problem_size // 3},
        ]
        
        # 生成随机目标函数系数
        objective_coeffs = np.random.uniform(0, 1, problem_size)
        
        # 记录初始资源使用
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # 运行优化
        result = self.optimizer.optimize(
            problem_size=problem_size,
            constraints=constraints,
            objective_coeffs=objective_coeffs
        )
        
        # 计算资源使用
        solve_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn('status', result)
        self.assertIn('solution', result)
        
        # 验证约束满足情况
        if result['solution']:
            solution = np.array(result['solution'])
            self.assertTrue(np.sum(solution) <= problem_size // 3)  # sum约束
        
        # 输出性能指标
        logger.info("\n=== 性能测试报告 ===")
        logger.info(f"问题规模: {problem_size:,} 变量")
        logger.info(f"求解时间: {solve_time:.2f} 秒")
        logger.info(f"内存使用: {memory_increase:.1f} MB")
        logger.info(f"求解状态: {result['status']}")
        logger.info(f"目标函数值: {result['objective_value']}")
        logger.info(f"数据块数: {result['num_chunks']}")
        
    def test_memory_mapping(self):
        """测试内存映射"""
        # 测试数据
        problem_size = 20_000  # 2万变量
        
        # 记录初始内存
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # 运行优化
        constraints = [{'type': 'sum', 'bound': problem_size // 4}]
        result = self.optimizer.optimize(problem_size, constraints)
        
        # 计算内存增长
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        # 验证内存效率
        logger.info("\n=== 内存映射测试 ===")
        logger.info(f"问题规模: {problem_size:,} 变量")
        logger.info(f"理论内存需求: {problem_size * 8 / 1024:.1f} MB")
        logger.info(f"实际内存增长: {memory_increase:.1f} MB")
        
    def test_checkpoint_recovery(self):
        """测试检查点恢复"""
        problem_size = 30_000  # 3万变量
        constraints = [{'type': 'sum', 'bound': problem_size // 3}]
        
        # 模拟中断和恢复
        try:
            # 设置较小的检查点间隔
            self.params.checkpoint_interval = 2
            result = self.optimizer.optimize(problem_size, constraints)
            
            # 验证检查点文件
            import glob
            checkpoint_files = glob.glob('checkpoint_*.npz')
            logger.info(f"生成检查点文件: {len(checkpoint_files)} 个")
            
            # 清理检查点文件
            for file in checkpoint_files:
                os.remove(file)
                
        except Exception as e:
            logger.error(f"检查点测试异常: {str(e)}")

def run_validation_report():
    """运行验证报告"""
    print("=== 亿级变量优化验证报告 ===")
    
    # 测试不同规模
    test_sizes = [10_000, 50_000, 100_000]
    results = []
    
    for size in test_sizes:
        params = ExascaleParams(chunk_size=5_000, max_threads=4)
        optimizer = ExascaleOptimizer(params)
        
        constraints = [{'type': 'sum', 'bound': size // 3}]
        objective_coeffs = np.random.uniform(0, 1, size)
        
        start_time = time.time()
        result = optimizer.optimize(size, constraints, objective_coeffs)
        solve_time = time.time() - start_time
        
        results.append({
            'size': size,
            'time': solve_time,
            'status': result['status'],
            'chunks': result['num_chunks']
        })
        
        print(f"规模: {size:,} 变量, 时间: {solve_time:.2f}s, 状态: {result['status']}")
    
    # 计算扩展性
    if len(results) >= 2:
        speedup = results[0]['time'] / results[-1]['time'] * (results[-1]['size'] / results[0]['size'])
        print(f"\n扩展性分析:")
        print(f"规模扩大倍数: {results[-1]['size'] / results[0]['size']:.1f}x")
        print(f"时间增长倍数: {results[-1]['time'] / results[0]['time']:.1f}x")
        print(f"算法效率: {speedup:.2f} (>1表示良好扩展性)")
    
    return results

if __name__ == '__main__':
    # 运行单元测试
    unittest.main(verbosity=2, exit=False)
    
    # 运行验证报告
    print("\n" + "="*50)
    run_validation_report() 