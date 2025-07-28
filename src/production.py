"""
生产决策优化模块
"""
# 尝试导入ortools，失败时使用备用方案
try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError as e:
    print(f"警告: OR-Tools导入失败: {e}，使用备用优化器")
    pywraplp = None
    HAS_ORTOOLS = False
from dataclasses import dataclass
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProductionParams:
    """生产参数"""
    defect_rate1: float
    defect_rate2: float
    test_cost1: float
    test_cost2: float
    assembly_cost: float
    test_cost_final: float
    repair_cost: float
    market_price: float
    return_loss: float
    
    def validate(self):
        """验证参数有效性"""
        if not (0 <= self.defect_rate1 <= 1 and 0 <= self.defect_rate2 <= 1):
            raise ValueError("不合格率必须在[0,1]范围内")
        if any(x < 0 for x in [self.test_cost1, self.test_cost2, self.assembly_cost,
                              self.test_cost_final, self.repair_cost,
                              self.market_price, self.return_loss]):
            raise ValueError("成本和价格必须为非负数")

class ProductionOptimizer:
    """生产决策优化器"""
    
    def __init__(self, params: ProductionParams):
        """初始化优化器
        
        Args:
            params: 生产参数
        """
        self.params = params
        self.params.validate()
        self.use_ortools = HAS_ORTOOLS
        
        if HAS_ORTOOLS:
            try:
                # 创建求解器（使用SCIP）
                self.solver = pywraplp.Solver.CreateSolver('SCIP')
                if not self.solver:
                    logger.warning("无法创建SCIP求解器，使用备用方案")
                    self.use_ortools = False
                else:
                    # 设置多线程
                    try:
                        # 简化SCIP参数设置
                        self.solver.SetSolverSpecificParametersAsString("parallel/maxnthreads=8")
                    except Exception as e:
                        logger.warning(f"设置多线程失败: {str(e)}")
                    
                    # 创建决策变量
                    self._build_model()
            except Exception as e:
                logger.warning(f"OR-Tools初始化失败: {e}，使用备用方案")
                self.use_ortools = False
        else:
            logger.info("OR-Tools不可用，使用备用优化方案")
        
    def _build_model(self):
        """构建优化模型"""
        # 决策变量
        self.x1 = self.solver.BoolVar('test_part1')  # 是否检测零件1
        self.x2 = self.solver.BoolVar('test_part2')  # 是否检测零件2
        self.y = self.solver.BoolVar('test_final')   # 是否检测成品
        self.z = self.solver.BoolVar('repair')       # 是否拆解返修
        
        # 辅助变量
        self.p_ok = self.solver.NumVar(0, 1, 'p_ok')  # 合格概率
        self.repair_cost = self.solver.NumVar(0, float('inf'), 'repair_cost')  # 返修成本
        self.return_loss = self.solver.NumVar(0, float('inf'), 'return_loss')  # 退货损失
        
        # 添加合格率约束
        p1 = self.params.defect_rate1
        p2 = self.params.defect_rate2
        
        # 线性化合格率约束
        # p_ok = (1 - p1(1-x1))(1 - p2(1-x2))
        # = 1 - p1(1-x1) - p2(1-x2) + p1p2(1-x1)(1-x2)
        self.solver.Add(self.p_ok <= 1)
        self.solver.Add(self.p_ok <= 1 - p1 * (1 - self.x1))
        self.solver.Add(self.p_ok <= 1 - p2 * (1 - self.x2))
        self.solver.Add(self.p_ok >= 1 - p1 * (1 - self.x1) - p2 * (1 - self.x2))
        
        # 线性化返修成本约束
        M = self.params.repair_cost  # 足够大的常数
        self.solver.Add(self.repair_cost >= 0)
        self.solver.Add(self.repair_cost <= M * self.z)  # 如果z=0，repair_cost=0
        self.solver.Add(self.repair_cost <= self.params.repair_cost * (1 - self.p_ok))
        self.solver.Add(self.repair_cost >= self.params.repair_cost * (1 - self.p_ok) - M * (1 - self.z))
        
        # 线性化退货损失约束
        M = self.params.return_loss  # 足够大的常数
        self.solver.Add(self.return_loss >= 0)
        self.solver.Add(self.return_loss <= M * (1 - self.z))  # 如果z=1，return_loss=0
        self.solver.Add(self.return_loss <= self.params.return_loss * (1 - self.p_ok))
        self.solver.Add(self.return_loss >= self.params.return_loss * (1 - self.p_ok) - M * self.z)
        
        # 目标函数：最大化期望利润
        revenue = self.params.market_price * self.p_ok
        total_cost = (
            self.params.test_cost1 * self.x1 +
            self.params.test_cost2 * self.x2 +
            self.params.assembly_cost +
            self.params.test_cost_final * self.y +
            self.repair_cost +
            self.return_loss
        )
        
        self.solver.Maximize(revenue - total_cost)
        
    def _fallback_heuristic(self) -> dict:
        """启发式求解（当优化求解失败时使用）
        
        Returns:
            dict: 启发式解
        """
        # 简单启发式：检测成本较低的零件
        test_part1 = self.params.test_cost1 < self.params.defect_rate1 * self.params.return_loss
        test_part2 = self.params.test_cost2 < self.params.defect_rate2 * self.params.return_loss
        test_final = False  # 成品检测成本较高，默认不检测
        repair = True      # 默认进行返修
        
        # 计算期望利润
        p_ok = (
            (1 - self.params.defect_rate1 * (1 - test_part1)) *
            (1 - self.params.defect_rate2 * (1 - test_part2))
        )
        
        revenue = self.params.market_price * p_ok
        total_cost = (
            self.params.test_cost1 * test_part1 +
            self.params.test_cost2 * test_part2 +
            self.params.assembly_cost +
            self.params.test_cost_final * test_final +
            self.params.repair_cost * repair * (1 - p_ok) +
            self.params.return_loss * (1 - p_ok) * (1 - repair)
        )
        
        return {
            'test_part1': test_part1,
            'test_part2': test_part2,
            'test_final': test_final,
            'repair': repair,
            'expected_profit': revenue - total_cost,
            'ok_probability': p_ok,
            'status': 'HEURISTIC',
            'solve_time': 0
        }
        
    def solve(self, timeout: int = 60) -> dict:
        """求解优化模型
        
        Args:
            timeout: 求解超时时间（秒）
            
        Returns:
            dict: 优化结果
        """
        logger.info("开始求解优化模型...")
        start_time = time.time()
        
        # 如果OR-Tools不可用，直接使用启发式方法
        if not self.use_ortools:
            logger.info("使用备用启发式优化算法")
            return self._fallback_heuristic()
        
        try:
            # 设置超时
            if timeout > 0:
                self.solver.SetTimeLimit(int(timeout * 1000))  # 转换为毫秒
            
            # 求解
            status = self.solver.Solve()
            solve_time = time.time() - start_time
            
            # 检查求解状态
            if status == pywraplp.Solver.OPTIMAL:
                # 提取结果
                result = {
                    'test_part1': bool(self.x1.solution_value()),
                    'test_part2': bool(self.x2.solution_value()),
                    'test_final': bool(self.y.solution_value()),
                    'repair': bool(self.z.solution_value()),
                    'expected_profit': self.solver.Objective().Value(),
                    'ok_probability': self.p_ok.solution_value(),
                    'status': 'OPTIMAL',
                    'solve_time': solve_time * 1000  # 转换为毫秒
                }
            else:
                # 使用启发式求解
                logger.warning("优化求解失败，使用启发式方法")
                result = self._fallback_heuristic()
            
            logger.info(f"求解完成，用时: {result['solve_time']:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"求解过程出错: {str(e)}")
            return self._fallback_heuristic()

def optimize_production(params: ProductionParams) -> dict:
    """优化生产决策
    
    Args:
        params: 生产参数
        
    Returns:
        dict: 优化结果
    """
    try:
        optimizer = ProductionOptimizer(params)
        return optimizer.solve()
    except Exception as e:
        logger.error(f"生产优化失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试优化器
    try:
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
        print("\n优化结果:")
        for key, value in result.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}") 