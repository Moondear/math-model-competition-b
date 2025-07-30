"""
多工序扩展模块
"""
import networkx as nx
# 尝试导入ortools，失败时使用备用方案
try:
    from ortools.linear_solver import pywraplp
    HAS_ORTOOLS = True
except ImportError as e:
    print(f"警告: OR-Tools导入失败: {e}，使用备用优化器")
    pywraplp = None
    HAS_ORTOOLS = False
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodeParams:
    """节点参数数据类"""
    defect_rate: float    # 不合格率
    process_cost: float   # 加工成本
    test_cost: float      # 检测成本
    repair_cost: float    # 返修成本
    
    def validate(self):
        """验证参数有效性"""
        if not 0 <= self.defect_rate <= 1:
            raise ValueError("不合格率必须在[0,1]范围内")
        if any(x < 0 for x in [self.process_cost, self.test_cost, self.repair_cost]):
            raise ValueError("所有成本必须为非负数")

class MultiStageOptimizer:
    """多工序优化器"""
    
    # 表2配置：8个零件、3个半成品、1个成品的完整参数
    TABLE2_CONFIG = {
        'components': {
            'C1': {'defect_rate': 0.10, 'purchase_cost': 2, 'test_cost': 1},
            'C2': {'defect_rate': 0.10, 'purchase_cost': 8, 'test_cost': 1},
            'C3': {'defect_rate': 0.10, 'purchase_cost': 12, 'test_cost': 2},
            'C4': {'defect_rate': 0.10, 'purchase_cost': 2, 'test_cost': 1},
            'C5': {'defect_rate': 0.10, 'purchase_cost': 8, 'test_cost': 1},
            'C6': {'defect_rate': 0.10, 'purchase_cost': 12, 'test_cost': 2},
            'C7': {'defect_rate': 0.10, 'purchase_cost': 8, 'test_cost': 1},
            'C8': {'defect_rate': 0.10, 'purchase_cost': 12, 'test_cost': 2}
        },
        'semi_products': {
            'SP1': {'defect_rate': 0.10, 'assembly_cost': 8, 'test_cost': 4, 'disassembly_cost': 6},
            'SP2': {'defect_rate': 0.10, 'assembly_cost': 8, 'test_cost': 4, 'disassembly_cost': 6},
            'SP3': {'defect_rate': 0.10, 'assembly_cost': 8, 'test_cost': 4, 'disassembly_cost': 6}
        },
        'final_product': {
            'FP': {'defect_rate': 0.10, 'assembly_cost': 8, 'test_cost': 6, 
                   'disassembly_cost': 10, 'market_price': 200, 'exchange_loss': 40}
        },
        'assembly_structure': {
            # 定义装配关系：哪些零件组装成哪个半成品/成品
            'SP1': ['C1', 'C2', 'C3'],  # 半成品1由零件1,2,3组装
            'SP2': ['C4', 'C5'],        # 半成品2由零件4,5组装  
            'SP3': ['C6', 'C7', 'C8'],  # 半成品3由零件6,7,8组装
            'FP': ['SP1', 'SP2', 'SP3'] # 成品由三个半成品组装
        }
    }
    
    @classmethod
    def load_table2_config(cls) -> nx.DiGraph:
        """加载表2配置构建生产网络
        
        Returns:
            nx.DiGraph: 根据表2配置构建的生产网络
        """
        logger.info("加载表2配置构建生产网络...")
        G = nx.DiGraph()
        
        # 添加零件节点
        for comp_id, comp_data in cls.TABLE2_CONFIG['components'].items():
            params = NodeParams(
                defect_rate=comp_data['defect_rate'],
                process_cost=comp_data['purchase_cost'],
                test_cost=comp_data['test_cost'],
                repair_cost=0  # 零件不需要返修，直接更换
            )
            G.add_node(comp_id, params=params, node_type='component')
        
        # 添加半成品节点
        for sp_id, sp_data in cls.TABLE2_CONFIG['semi_products'].items():
            params = NodeParams(
                defect_rate=sp_data['defect_rate'],
                process_cost=sp_data['assembly_cost'],
                test_cost=sp_data['test_cost'],
                repair_cost=sp_data['disassembly_cost']
            )
            G.add_node(sp_id, params=params, node_type='semi_product')
        
        # 添加成品节点
        for fp_id, fp_data in cls.TABLE2_CONFIG['final_product'].items():
            params = NodeParams(
                defect_rate=fp_data['defect_rate'],
                process_cost=fp_data['assembly_cost'],
                test_cost=fp_data['test_cost'],
                repair_cost=fp_data['disassembly_cost']
            )
            # 添加市场价格和调换损失属性
            G.add_node(fp_id, params=params, node_type='final_product',
                      market_price=fp_data['market_price'],
                      exchange_loss=fp_data['exchange_loss'])
        
        # 添加装配关系边
        for product_id, input_ids in cls.TABLE2_CONFIG['assembly_structure'].items():
            for input_id in input_ids:
                G.add_edge(input_id, product_id)
        
        logger.info(f"生产网络构建完成: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
        return G
    
    @classmethod
    def create_custom_network(cls, custom_config: Dict) -> nx.DiGraph:
        """根据自定义配置创建生产网络
        
        Args:
            custom_config: 自定义配置字典，格式与TABLE2_CONFIG相同
            
        Returns:
            nx.DiGraph: 自定义生产网络
        """
        logger.info("根据自定义配置创建生产网络...")
        
        # 临时替换配置
        original_config = cls.TABLE2_CONFIG
        cls.TABLE2_CONFIG = custom_config
        
        try:
            network = cls.load_table2_config()
            return network
        finally:
            # 恢复原配置
            cls.TABLE2_CONFIG = original_config
    
    def __init__(self, graph: nx.DiGraph):
        """初始化优化器
        
        Args:
            graph: networkx有向图，表示生产网络
                  节点属性：params (NodeParams)
                  边属性：无
        """
        try:
            self.graph = graph
            self._validate_graph()
            
            # 初始化求解器
            self.solver = pywraplp.Solver.CreateSolver('SCIP')
            if not self.solver:
                raise ValueError("求解器创建失败，请确保已安装ortools")
                
            self.solver.SetNumThreads(8)  # 启用多核
            self._build_model()
            
        except Exception as e:
            logger.error(f"优化器初始化失败: {str(e)}")
            raise
            
    def _validate_graph(self):
        """验证图结构的有效性"""
        # 检查是否是有向无环图
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("生产网络必须是有向无环图")
            
        # 检查节点参数
        for node in self.graph.nodes:
            if 'params' not in self.graph.nodes[node]:
                raise ValueError(f"节点 {node} 缺少参数")
            self.graph.nodes[node]['params'].validate()
            
    def _build_model(self):
        """构建优化模型"""
        try:
            # 为每个节点创建决策变量
            self.test_vars = {}    # 是否检测
            self.repair_vars = {}  # 是否返修
            self.p_ok_vars = {}    # 合格率
            self.cost_vars = {}    # 总成本
            
            for node in self.graph.nodes:
                # 检测决策
                self.test_vars[node] = self.solver.BoolVar(f'test_{node}')
                # 返修决策
                self.repair_vars[node] = self.solver.BoolVar(f'repair_{node}')
                # 合格率
                self.p_ok_vars[node] = self.solver.NumVar(0, 1, f'p_ok_{node}')
                # 总成本
                self.cost_vars[node] = self.solver.NumVar(0, float('inf'), f'cost_{node}')
            
            # 添加约束
            for node in nx.topological_sort(self.graph):
                self._add_node_constraints(node)
            
            # 构建目标函数
            final_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
            if len(final_nodes) != 1:
                raise ValueError("必须有且仅有一个终点节点")
                
            final_node = final_nodes[0]
            self.solver.Minimize(self.cost_vars[final_node])
            
        except Exception as e:
            logger.error(f"模型构建失败: {str(e)}")
            raise
            
    def _add_node_constraints(self, node: str):
        """添加节点相关的约束
        
        Args:
            node: 节点名称
        """
        params = self.graph.nodes[node]['params']
        pred = list(self.graph.predecessors(node))
        
        # 1. 合格率约束
        if not pred:  # 起始节点
            # P_ok = 1 - p_defect * (1 - test)
            self.solver.Add(self.p_ok_vars[node] <= 1)
            self.solver.Add(self.p_ok_vars[node] <= 1 - params.defect_rate * 
                          (1 - self.test_vars[node]))
            self.solver.Add(self.p_ok_vars[node] >= 1 - params.defect_rate)
        else:  # 中间节点
            # P_ok = prod(P_ok_pred) * (1 - p_defect * (1 - test))
            # 使用对数线性化
            log_p_ok = self.solver.NumVar(float('-inf'), 0, f'log_p_ok_{node}')
            M = len(pred) + 1  # 足够大的常数
            
            # log(P_ok) = sum(log(P_ok_pred)) + log(1 - p_defect * (1 - test))
            self.solver.Add(log_p_ok <= 0)
            for p in pred:
                self.solver.Add(log_p_ok <= self.p_ok_vars[p])
            self.solver.Add(log_p_ok <= 1 - params.defect_rate * 
                          (1 - self.test_vars[node]))
            
            # 转回原始概率
            self.solver.Add(self.p_ok_vars[node] >= 0)
            self.solver.Add(self.p_ok_vars[node] <= M * (1 + log_p_ok))
        
        # 2. 成本约束
        # 处理成本 = 加工成本 + 检测成本 * test + 返修成本 * repair * (1-P_ok)
        process_cost = params.process_cost
        test_cost = params.test_cost * self.test_vars[node]
        
        # 线性化返修成本
        repair_cost_var = self.solver.NumVar(0, float('inf'), f'repair_cost_{node}')
        M = params.repair_cost  # 足够大的常数
        self.solver.Add(repair_cost_var >= 0)
        self.solver.Add(repair_cost_var <= M * self.repair_vars[node])
        self.solver.Add(repair_cost_var <= params.repair_cost * (1 - self.p_ok_vars[node]))
        self.solver.Add(repair_cost_var >= params.repair_cost * (1 - self.p_ok_vars[node]) - 
                       M * (1 - self.repair_vars[node]))
        
        # 总成本 = 当前节点成本 + 前驱节点成本之和
        pred_costs = sum(self.cost_vars[p] for p in pred) if pred else 0
        self.solver.Add(self.cost_vars[node] == process_cost + test_cost + 
                       repair_cost_var + pred_costs)
        
        # 3. 逻辑约束
        # 只有检测了才能返修
        self.solver.Add(self.repair_vars[node] <= self.test_vars[node])
        
    def solve(self, timeout: int = 60) -> Dict:
        """求解优化模型
        
        Args:
            timeout: 求解超时时间(秒)
            
        Returns:
            Dict: 求解结果
        """
        try:
            start_time = time.time()
            
            # 设置求解时限
            self.solver.SetTimeLimit(timeout * 1000)
            
            # 求解模型
            logger.info("开始求解优化模型...")
            status = self.solver.Solve()
            
            # 检查求解状态
            if status != pywraplp.Solver.OPTIMAL:
                logger.warning(f"精确求解失败(状态码:{status})，切换到启发式算法")
                return self._fallback_heuristic()
                
            # 提取结果
            solution = {
                'decisions': {},
                'total_cost': None,
                'is_optimal': True,
                'solver_status': 'OPTIMAL',
                'solution_time': time.time() - start_time
            }
            
            # 收集每个节点的决策
            for node in self.graph.nodes:
                solution['decisions'][node] = {
                    'test': bool(self.test_vars[node].solution_value()),
                    'repair': bool(self.repair_vars[node].solution_value()),
                    'p_ok': self.p_ok_vars[node].solution_value(),
                    'cost': self.cost_vars[node].solution_value()
                }
            
            # 获取总成本
            final_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
            solution['total_cost'] = self.cost_vars[final_nodes[0]].solution_value()
            
            logger.info(f"求解完成，用时: {solution['solution_time']*1000:.2f}ms")
            return solution
            
        except Exception as e:
            logger.error(f"求解过程出错: {str(e)}")
            return self._fallback_heuristic()
            
    def _fallback_heuristic(self) -> Dict:
        """启发式算法作为备选方案
        
        当精确求解失败时使用简单的启发式规则
        """
        try:
            solution = {
                'decisions': {},
                'total_cost': 0,
                'is_optimal': False,
                'solver_status': 'HEURISTIC'
            }
            
            # 简单规则：
            # 1. 高不合格率的节点必检
            # 2. 检测的节点允许返修
            # 3. 按拓扑序计算成本和合格率
            for node in nx.topological_sort(self.graph):
                params = self.graph.nodes[node]['params']
                pred = list(self.graph.predecessors(node))
                
                # 决策规则
                test = params.defect_rate > 0.15
                repair = test
                
                # 计算合格率
                if not pred:  # 起始节点
                    p_ok = 1 - params.defect_rate * (1 - test)
                else:  # 中间节点
                    pred_p_ok = 1
                    for p in pred:
                        pred_p_ok *= solution['decisions'][p]['p_ok']
                    p_ok = pred_p_ok * (1 - params.defect_rate * (1 - test))
                
                # 计算成本
                process_cost = params.process_cost
                test_cost = params.test_cost * test
                repair_cost = params.repair_cost * repair * (1 - p_ok)
                pred_costs = sum(solution['decisions'][p]['cost'] for p in pred) if pred else 0
                total_cost = process_cost + test_cost + repair_cost + pred_costs
                
                # 保存结果
                solution['decisions'][node] = {
                    'test': test,
                    'repair': repair,
                    'p_ok': p_ok,
                    'cost': total_cost
                }
            
            # 获取总成本
            final_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
            solution['total_cost'] = solution['decisions'][final_nodes[0]]['cost']
            
            return solution
            
        except Exception as e:
            logger.error(f"启发式算法失败: {str(e)}")
            raise

def create_example_network() -> nx.DiGraph:
    """创建示例生产网络
    
    Returns:
        nx.DiGraph: 示例生产网络
    """
    G = nx.DiGraph()
    
    # 添加节点
    nodes = {
        # 零件节点
        'P1': NodeParams(0.1, 4, 2, 5),   # 零件1
        'P2': NodeParams(0.1, 4, 2, 5),   # 零件2
        'P3': NodeParams(0.1, 4, 2, 5),   # 零件3
        'P4': NodeParams(0.1, 4, 2, 5),   # 零件4
        'P5': NodeParams(0.1, 4, 2, 5),   # 零件5
        'P6': NodeParams(0.1, 4, 2, 5),   # 零件6
        
        # 半成品节点
        'A1': NodeParams(0.1, 6, 3, 8),   # 半成品1
        'A2': NodeParams(0.1, 6, 3, 8),   # 半成品2
        'A3': NodeParams(0.1, 6, 3, 8),   # 半成品3
        
        # 成品节点
        'F': NodeParams(0.05, 8, 4, 10)   # 成品
    }
    
    for node, params in nodes.items():
        G.add_node(node, params=params)
    
    # 添加边
    edges = [
        # 零件到半成品
        ('P1', 'A1'),  # 半成品1由零件1和2组装
        ('P2', 'A1'),
        ('P3', 'A2'),  # 半成品2由零件3和4组装
        ('P4', 'A2'),
        ('P5', 'A3'),  # 半成品3由零件5和6组装
        ('P6', 'A3'),
        
        # 半成品到成品
        ('A1', 'F'),   # 成品由三个半成品组装
        ('A2', 'F'),
        ('A3', 'F')
    ]
    
    G.add_edges_from(edges)
    
    return G

def optimize_multistage(graph: nx.DiGraph) -> Dict:
    """优化多工序生产系统
    
    Args:
        graph: 生产网络
        
    Returns:
        Dict: 优化结果
    """
    try:
        optimizer = MultiStageOptimizer(graph)
        return optimizer.solve()
    except Exception as e:
        logger.error(f"多工序优化失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试优化器
    try:
        # 创建示例网络
        graph = create_example_network()
        
        # 运行优化
        result = optimize_multistage(graph)
        
        # 输出结果
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
            
    except Exception as e:
        logger.error(f"测试失败: {str(e)}") 