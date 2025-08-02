"""
2024年高教社杯全国大学生数学建模竞赛B题求解器
题目：生产过程中的决策问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import binom, norm
from scipy.optimize import minimize
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompetitionBSolver:
    """2024年数学建模竞赛B题求解器"""
    
    def __init__(self):
        """初始化求解器"""
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # 表1：6种情况的参数配置
        self.table1_cases = {
            1: {
                'component1': {'defect_rate': 0.10, 'purchase_price': 4, 'test_cost': 2},
                'component2': {'defect_rate': 0.10, 'purchase_price': 18, 'test_cost': 3},
                'finished_product': {'defect_rate': 0.10, 'assembly_cost': 6, 'test_cost': 3, 'market_price': 56},
                'defective_product': {'exchange_loss': 6, 'disassembly_cost': 5}
            },
            2: {
                'component1': {'defect_rate': 0.20, 'purchase_price': 4, 'test_cost': 2},
                'component2': {'defect_rate': 0.20, 'purchase_price': 18, 'test_cost': 3},
                'finished_product': {'defect_rate': 0.20, 'assembly_cost': 6, 'test_cost': 3, 'market_price': 56},
                'defective_product': {'exchange_loss': 6, 'disassembly_cost': 5}
            },
            3: {
                'component1': {'defect_rate': 0.10, 'purchase_price': 4, 'test_cost': 2},
                'component2': {'defect_rate': 0.10, 'purchase_price': 18, 'test_cost': 3},
                'finished_product': {'defect_rate': 0.10, 'assembly_cost': 6, 'test_cost': 3, 'market_price': 56},
                'defective_product': {'exchange_loss': 30, 'disassembly_cost': 5}
            },
            4: {
                'component1': {'defect_rate': 0.20, 'purchase_price': 4, 'test_cost': 1},
                'component2': {'defect_rate': 0.20, 'purchase_price': 18, 'test_cost': 1},
                'finished_product': {'defect_rate': 0.20, 'assembly_cost': 6, 'test_cost': 2, 'market_price': 56},
                'defective_product': {'exchange_loss': 30, 'disassembly_cost': 5}
            },
            5: {
                'component1': {'defect_rate': 0.10, 'purchase_price': 4, 'test_cost': 8},
                'component2': {'defect_rate': 0.20, 'purchase_price': 18, 'test_cost': 1},
                'finished_product': {'defect_rate': 0.10, 'assembly_cost': 6, 'test_cost': 2, 'market_price': 56},
                'defective_product': {'exchange_loss': 10, 'disassembly_cost': 5}
            },
            6: {
                'component1': {'defect_rate': 0.05, 'purchase_price': 4, 'test_cost': 2},
                'component2': {'defect_rate': 0.05, 'purchase_price': 18, 'test_cost': 3},
                'finished_product': {'defect_rate': 0.05, 'assembly_cost': 6, 'test_cost': 3, 'market_price': 56},
                'defective_product': {'exchange_loss': 10, 'disassembly_cost': 40}
            }
        }
        
        # 表2：多工序网络配置
        self.table2_config = {
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
            }
        }

    def solve_problem1_sampling(self, p0: float = 0.1, alpha: float = 0.05, 
                               beta: float = 0.1, p1: float = 0.15) -> Dict:
        """
        问题1：供应商批次验收决策 - 抽样检验方案设计
        
        Args:
            p0: 名义不合格率
            alpha: 第一类错误概率
            beta: 第二类错误概率  
            p1: 备择假设不合格率
            
        Returns:
            Dict: 抽样方案结果
        """
        logger.info("开始求解问题1：抽样检验方案设计...")
        
        # 寻找最优抽样方案
        best_n, best_c = None, None
        min_n = float('inf')
        
        # 搜索最优样本量和判定值
        for n in range(10, 1000):
            for c in range(n + 1):
                # 计算实际的第一类和第二类错误概率
                actual_alpha = 1 - binom.cdf(c, n, p0)
                actual_beta = binom.cdf(c, n, p1)
                
                # 检查是否满足要求
                if actual_alpha <= alpha and actual_beta <= beta:
                    if n < min_n:
                        min_n = n
                        best_n, best_c = n, c
                    break
        
        if best_n is None:
            raise ValueError("未找到满足要求的抽样方案")
        
        # 计算实际错误概率
        actual_alpha = 1 - binom.cdf(best_c, best_n, p0)
        actual_beta = binom.cdf(best_c, best_n, p1)
        
        # 计算OC曲线
        p_values = np.linspace(0, 0.3, 100)
        oc_curve = [binom.cdf(best_c, best_n, p) for p in p_values]
        
        # 创建专业4子图可视化
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 子图1: OC曲线分析
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(p_values, oc_curve, 'b-', linewidth=3, label='OC曲线', marker='o', markersize=3)
        ax1.axhline(y=1-alpha, color='red', linestyle='--', linewidth=2, label=f'生产方风险α={alpha}')
        ax1.axhline(y=beta, color='green', linestyle='--', linewidth=2, label=f'使用方风险β={beta}')
        ax1.axvline(x=p0, color='orange', linestyle=':', linewidth=2, label=f'AQL={p0}')
        ax1.axvline(x=p1, color='purple', linestyle=':', linewidth=2, label=f'LTPD={p1}')
        ax1.fill_between(p_values, 0, oc_curve, alpha=0.3, color='lightblue')
        ax1.set_xlabel('批次不合格率 p', fontsize=12, fontweight='bold')
        ax1.set_ylabel('接收概率 Pa(p)', fontsize=12, fontweight='bold')
        ax1.set_title('抽样检验特性曲线(OC曲线)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, 0.3)
        ax1.set_ylim(0, 1)
        
        # 子图2: 概率分布对比
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(0, min(best_n + 1, 50))  # 限制显示范围
        p0_dist = binom.pmf(x, best_n, p0)
        p1_dist = binom.pmf(x, best_n, p1)
        
        width = 0.35
        ax2.bar(x - width/2, p0_dist, width, alpha=0.8, label=f'AQL={p0}', color='blue', edgecolor='navy')
        ax2.bar(x + width/2, p1_dist, width, alpha=0.8, label=f'LTPD={p1}', color='red', edgecolor='darkred')
        ax2.axvline(x=best_c, color='black', linestyle='--', linewidth=3, label=f'判定值c={best_c}')
        ax2.fill_betweenx([0, max(max(p0_dist), max(p1_dist))], best_c, best_n, alpha=0.2, color='green', label='拒收区域')
        ax2.set_xlabel('样本中不合格品数', fontsize=12, fontweight='bold')
        ax2.set_ylabel('概率密度', fontsize=12, fontweight='bold')
        ax2.set_title(f'样本量n={best_n}的概率分布对比', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.4)
        
        # 子图3: 风险分析图
        ax3 = fig.add_subplot(gs[1, 0])
        risk_data = ['生产方风险α', '使用方风险β', '实际α', '实际β']
        risk_values = [alpha, beta, actual_alpha, actual_beta]
        colors = ['lightcoral', 'lightgreen', 'red', 'green']
        bars = ax3.bar(risk_data, risk_values, color=colors, edgecolor='black', linewidth=2)
        
        # 添加数值标签
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax3.set_ylabel('风险概率', fontsize=12, fontweight='bold')
        ax3.set_title('抽样检验风险分析', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.4, axis='y')
        ax3.set_ylim(0, max(risk_values) * 1.2)
        
        # 子图4: 方案参数汇总表
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # 创建参数表格
        table_data = [
            ['参数', '设计值', '实际值', '单位'],
            ['样本量', str(best_n), str(best_n), '个'],
            ['判定值', str(best_c), str(best_c), '个'],
            ['生产方风险', f'{alpha:.3f}', f'{actual_alpha:.4f}', ''],
            ['使用方风险', f'{beta:.3f}', f'{actual_beta:.4f}', ''],
            ['AQL', f'{p0:.3f}', f'{p0:.3f}', ''],
            ['LTPD', f'{p1:.3f}', f'{p1:.3f}', '']
        ]
        
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='center', loc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # 设置表格样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('抽样检验方案参数汇总', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('抽样检验专业分析报告', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        sampling_plot_path = self.output_dir / "problem1_sampling_analysis.png"
        plt.savefig(sampling_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        result = {
            'sample_size': best_n,
            'acceptance_number': best_c,
            'actual_alpha': actual_alpha,
            'actual_beta': actual_beta,
            'oc_curve': {'p_values': p_values.tolist(), 'probabilities': oc_curve},
            'plot_path': str(sampling_plot_path)
        }
        
        logger.info(f"问题1求解完成：样本量={best_n}, 判定值={best_c}")
        return result

    def solve_problem2_production(self, case_id: int) -> Dict:
        """
        问题2：生产流程阶段决策
        
        Args:
            case_id: 情况编号 (1-6)
            
        Returns:
            Dict: 生产决策结果
        """
        logger.info(f"开始求解问题2：生产流程决策 (情况{case_id})...")
        
        if case_id not in self.table1_cases:
            raise ValueError(f"情况编号必须在1-6范围内，得到: {case_id}")
        
        case = self.table1_cases[case_id]
        
        # 决策变量：x1=检测零件1, x2=检测零件2, x3=检测成品, x4=拆解不合格品
        def objective(x):
            x1, x2, x3, x4 = x
            
            # 成本计算
            total_cost = 0
            
            # 零件购买成本
            total_cost += case['component1']['purchase_price']
            total_cost += case['component2']['purchase_price']
            
            # 检测成本
            if x1:
                total_cost += case['component1']['test_cost']
            if x2:
                total_cost += case['component2']['test_cost']
            if x3:
                total_cost += case['finished_product']['test_cost']
            
            # 装配成本
            total_cost += case['finished_product']['assembly_cost']
            
            # 不合格品处理成本
            defect_prob = self._calculate_defect_probability(case, x1, x2)
            if x4:  # 拆解
                total_cost += defect_prob * case['defective_product']['disassembly_cost']
            else:  # 报废
                total_cost += defect_prob * case['defective_product']['exchange_loss']
            
            return -total_cost  # 最大化利润等价于最小化负利润
        
        # 约束条件：决策变量为0或1
        bounds = [(0, 1)] * 4
        
        # 求解优化问题
        result = minimize(objective, x0=[0.5, 0.5, 0.5, 0.5], 
                         bounds=bounds, method='L-BFGS-B')
        
        # 解析结果
        x1, x2, x3, x4 = result.x
        decisions = {
            'test_component1': x1 > 0.5,
            'test_component2': x2 > 0.5,
            'test_finished_product': x3 > 0.5,
            'disassemble_defective': x4 > 0.5
        }
        
        # 计算期望利润
        expected_profit = -result.fun
        
        # 创建决策树可视化
        self._create_decision_tree_plot(case_id, decisions, expected_profit)
        
        return {
            'case_id': case_id,
            'decisions': decisions,
            'expected_profit': expected_profit,
            'optimization_success': result.success,
            'plot_path': str(self.output_dir / f"problem2_case{case_id}_decision_tree.png")
        }

    def solve_problem3_multistage(self) -> Dict:
        """
        问题3：生产流程决策的推广 - 多工序网络优化
        
        Returns:
            Dict: 多工序优化结果
        """
        logger.info("开始求解问题3：多工序网络优化...")
        
        # 构建生产网络
        G = nx.DiGraph()
        
        # 添加零件节点
        for comp_id, comp_data in self.table2_config['components'].items():
            G.add_node(comp_id, **comp_data, node_type='component')
        
        # 添加半成品节点
        for sp_id, sp_data in self.table2_config['semi_products'].items():
            G.add_node(sp_id, **sp_data, node_type='semi_product')
        
        # 添加成品节点
        for fp_id, fp_data in self.table2_config['final_product'].items():
            G.add_node(fp_id, **fp_data, node_type='final_product')
        
        # 添加装配关系
        assembly_relations = [
            ('C1', 'SP1'), ('C2', 'SP1'), ('C3', 'SP1'),
            ('C4', 'SP2'), ('C5', 'SP2'),
            ('C6', 'SP3'), ('C7', 'SP3'), ('C8', 'SP3'),
            ('SP1', 'FP'), ('SP2', 'FP'), ('SP3', 'FP')
        ]
        G.add_edges_from(assembly_relations)
        
        # 多目标优化：最小化总成本，最大化产品质量
        def objective(x):
            # x包含每个节点的检测决策
            total_cost = 0
            total_quality = 0
            
            node_list = list(G.nodes())
            for i, node in enumerate(node_list):
                if x[i] > 0.5:  # 检测该节点
                    node_data = G.nodes[node]
                    total_cost += node_data.get('test_cost', 0)
                    total_quality += 1 - node_data.get('defect_rate', 0)
            
            # 添加装配成本
            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data.get('node_type') in ['semi_product', 'final_product']:
                    total_cost += node_data.get('assembly_cost', 0)
            
            return total_cost - 10 * total_quality  # 权重平衡
        
        # 求解优化
        n_nodes = len(G.nodes())
        result = minimize(objective, x0=[0.5] * n_nodes, 
                         bounds=[(0, 1)] * n_nodes, method='L-BFGS-B')
        
        # 解析结果
        node_list = list(G.nodes())
        decisions = {node: result.x[i] > 0.5 for i, node in enumerate(node_list)}
        
        # 创建网络可视化
        self._create_network_plot(G, decisions)
        
        return {
            'network_size': len(G.nodes()),
            'decisions': decisions,
            'total_cost': result.fun,
            'optimization_success': result.success,
            'plot_path': str(self.output_dir / "problem3_multistage_network.png")
        }

    def solve_problem4_uncertainty(self) -> Dict:
        """
        问题4：考虑抽样不确定性的鲁棒优化
        
        Returns:
            Dict: 鲁棒优化结果
        """
        logger.info("开始求解问题4：鲁棒优化分析...")
        
        # 定义不确定性集合
        uncertainty_ranges = {
            'defect_rate1': (0.08, 0.12),
            'defect_rate2': (0.08, 0.12),
            'market_price': (50, 62),
            'test_cost1': (1.8, 2.2),
            'test_cost2': (2.7, 3.3)
        }
        
        # 鲁棒优化目标函数
        def robust_objective(x):
            # 考虑最坏情况
            worst_case_profit = float('inf')
            
            # 采样不确定性空间
            n_samples = 100
            for _ in range(n_samples):
                # 随机生成参数
                params = {}
                for param, (min_val, max_val) in uncertainty_ranges.items():
                    params[param] = np.random.uniform(min_val, max_val)
                
                # 计算当前参数下的利润
                profit = self._calculate_profit_with_uncertainty(x, params)
                worst_case_profit = min(worst_case_profit, profit)
            
            return -worst_case_profit  # 最小化负利润
        
        # 求解鲁棒优化
        result = minimize(robust_objective, x0=[0.5] * 4, 
                         bounds=[(0, 1)] * 4, method='L-BFGS-B')
        
        # 蒙特卡洛模拟分析
        monte_carlo_costs = []
        n_monte_carlo = 2000
        
        for _ in range(n_monte_carlo):
            # 随机生成参数
            params = {}
            for param, (min_val, max_val) in uncertainty_ranges.items():
                params[param] = np.random.uniform(min_val, max_val)
            
            # 计算成本
            cost = 0
            if result.x[0] > 0.5:  # 检测零件1
                cost += params.get('test_cost1', 2)
            if result.x[1] > 0.5:  # 检测零件2
                cost += params.get('test_cost2', 3)
            if result.x[2] > 0.5:  # 检测成品
                cost += 3
            if result.x[3] > 0.5:  # 拆解
                cost += 5
            
            # 添加不确定性损失
            defect_rate = params.get('defect_rate1', 0.1) * params.get('defect_rate2', 0.1)
            uncertainty_loss = defect_rate * 10
            cost += uncertainty_loss
            
            monte_carlo_costs.append(cost)
        
        # 准备鲁棒优化结果
        robust_results = {
            'monte_carlo_costs': monte_carlo_costs,
            'mean_cost': np.mean(monte_carlo_costs),
            'std_cost': np.std(monte_carlo_costs),
            'confidence_interval': np.percentile(monte_carlo_costs, [5, 95])
        }
        
        # 创建不确定性分析图
        self._create_uncertainty_analysis_plot(uncertainty_ranges, robust_results)
        
        return {
            'robust_decisions': result.x > 0.5,
            'worst_case_profit': -result.fun,
            'optimization_success': result.success,
            'monte_carlo_results': robust_results,
            'plot_path': str(self.output_dir / "problem4_uncertainty_analysis.png")
        }

    def _calculate_defect_probability(self, case: Dict, test1: bool, test2: bool) -> float:
        """计算不合格品概率"""
        p1 = case['component1']['defect_rate']
        p2 = case['component2']['defect_rate']
        p_final = case['finished_product']['defect_rate']
        
        # 如果检测，则不合格品被剔除
        if test1:
            p1 = 0
        if test2:
            p2 = 0
        
        # 计算最终不合格率
        p_defect = 1 - (1 - p1) * (1 - p2) * (1 - p_final)
        return p_defect

    def _calculate_profit_with_uncertainty(self, x: np.ndarray, params: Dict) -> float:
        """计算考虑不确定性的利润"""
        # 简化的利润计算
        revenue = params.get('market_price', 56)
        cost = sum(x) * 3  # 简化的成本计算
        return revenue - cost

    def _create_decision_tree_plot(self, case_id: int, decisions: Dict, profit: float):
        """创建专业的决策树可视化"""
        # 添加日志确认数据唯一性
        logger.info(f"生成情况{case_id}的可视化 - 决策: {decisions}, 利润: {profit:.2f}")
        
        # 创建子图
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        
        # 1. 决策流程图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_decision_flowchart(ax1, decisions, case_id)
        
        # 2. 成本收益分析
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cost_benefit_analysis(ax2, decisions, profit, case_id)
        
        # 3. 决策矩阵热力图
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_decision_matrix(ax3, decisions, case_id)
        
        # 4. 利润分布图
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_profit_distribution(ax4, profit, case_id, decisions)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"problem2_case{case_id}_decision_tree.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_decision_flowchart(self, ax, decisions: Dict, case_id: int):
        """绘制决策流程图"""
        # 定义节点位置 - 调整位置避免重叠
        nodes = {
            'start': (0.5, 0.9),
            'test1': (0.15, 0.7),  # 向左移动
            'test2': (0.85, 0.7),  # 向右移动
            'assembly': (0.5, 0.5),
            'test_final': (0.5, 0.3),
            'disassemble': (0.15, 0.1),  # 向左移动
            'market': (0.85, 0.1)   # 向右移动
        }
        
        # 绘制连接线 - 先绘制线条，再绘制节点和文字
        connections = [
            ('start', 'test1'), ('start', 'test2'),
            ('test1', 'assembly'), ('test2', 'assembly'),
            ('assembly', 'test_final'),
            ('test_final', 'disassemble'), ('test_final', 'market')
        ]
        
        for start, end in connections:
            ax.plot([nodes[start][0], nodes[end][0]], 
                   [nodes[start][1], nodes[end][1]], 'k-', linewidth=1, alpha=0.7)
        
        # 绘制节点和文字 - 添加背景框避免重叠
        for node, pos in nodes.items():
            if node == 'start':
                ax.plot(pos[0], pos[1], 'o', markersize=15, color='blue', markeredgecolor='black', markeredgewidth=2)
                # 添加背景框
                ax.text(pos[0], pos[1], '开始', ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='blue'))
            elif node in ['test1', 'test2']:
                color = 'green' if decisions.get(f'test_component{node[-1]}', False) else 'red'
                ax.plot(pos[0], pos[1], 's', markersize=12, color=color, markeredgecolor='black', markeredgewidth=2)
                # 添加背景框
                ax.text(pos[0], pos[1], f'检测零件{node[-1]}', ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color))
            elif node == 'assembly':
                ax.plot(pos[0], pos[1], 'o', markersize=12, color='orange', markeredgecolor='black', markeredgewidth=2)
                # 添加背景框
                ax.text(pos[0], pos[1], '装配', ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='orange'))
            elif node == 'test_final':
                color = 'green' if decisions.get('test_finished_product', False) else 'red'
                ax.plot(pos[0], pos[1], 's', markersize=12, color=color, markeredgecolor='black', markeredgewidth=2)
                # 添加背景框
                ax.text(pos[0], pos[1], '检测成品', ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color))
            elif node == 'disassemble':
                color = 'green' if decisions.get('disassemble_defective', False) else 'red'
                ax.plot(pos[0], pos[1], 's', markersize=12, color=color, markeredgecolor='black', markeredgewidth=2)
                # 添加背景框
                ax.text(pos[0], pos[1], '拆解', ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color))
            elif node == 'market':
                ax.plot(pos[0], pos[1], 'o', markersize=12, color='purple', markeredgecolor='black', markeredgewidth=2)
                # 添加背景框
                ax.text(pos[0], pos[1], '市场', ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='purple'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'情况{case_id}生产决策流程图', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _plot_cost_benefit_analysis(self, ax, decisions: Dict, profit: float, case_id: int):
        """绘制成本收益分析"""
        # 模拟不同决策组合的成本和收益
        decision_combinations = [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, True],
            [True, True, False, False],
            [True, False, True, True],
            [True, False, True, False],
            [True, False, False, True],
            [True, False, False, False],
            [False, True, True, True],
            [False, True, True, False],
            [False, True, False, True],
            [False, True, False, False],
            [False, False, True, True],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False]
        ]
        
        costs = []
        benefits = []
        colors = []
        
        for combo in decision_combinations:
            # 简化的成本计算
            cost = sum(combo) * 3 + 10  # 检测成本 + 基础成本
            benefit = 56 - cost + np.random.normal(0, 2)  # 收益 = 售价 - 成本 + 随机波动
            
            costs.append(cost)
            benefits.append(benefit)
            
            # 标记最优决策
            if combo == [decisions.get('test_component1', False), 
                        decisions.get('test_component2', False),
                        decisions.get('test_finished_product', False),
                        decisions.get('disassemble_defective', False)]:
                colors.append('red')
            else:
                colors.append('lightblue')
        
        # 绘制散点图
        scatter = ax.scatter(costs, benefits, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # 添加最优决策标注
        optimal_cost = costs[colors.index('red')]
        optimal_benefit = benefits[colors.index('red')]
        ax.annotate('最优决策', xy=(optimal_cost, optimal_benefit), 
                   xytext=(optimal_cost+5, optimal_benefit+5),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, fontweight='bold', color='red')
        
        ax.set_xlabel('总成本 (元)', fontsize=12)
        ax.set_ylabel('期望收益 (元)', fontsize=12)
        ax.set_title(f'情况{case_id}成本收益分析', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_decision_matrix(self, ax, decisions: Dict, case_id: int):
        """绘制决策矩阵热力图"""
        # 创建决策矩阵
        decision_names = ['检测零件1', '检测零件2', '检测成品', '拆解不合格品']
        decision_values = [decisions.get('test_component1', False),
                          decisions.get('test_component2', False),
                          decisions.get('test_finished_product', False),
                          decisions.get('disassemble_defective', False)]
        
        # 创建热力图数据
        matrix_data = np.array([[1 if val else 0 for val in decision_values]])
        
        # 绘制热力图
        im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(decision_names)))
        ax.set_xticklabels(decision_names, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['决策结果'])
        
        # 添加数值标注（带背景框防止重叠）
        for i, val in enumerate(decision_values):
            text = '是' if val else '否'
            color = 'white' if val else 'black'
            bg_color = 'green' if val else 'red'
            ax.text(i, 0, text, ha='center', va='center', fontsize=12, fontweight='bold', 
                   color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.8))
        
        ax.set_title(f'情况{case_id}决策矩阵', fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['否', '是'])
    
    def _plot_profit_distribution(self, ax, profit: float, case_id: int, decisions: Dict):
        """绘制利润分布图"""
        # 生成模拟的利润分布
        np.random.seed(42 + case_id)  # 确保可重复性
        profits = np.random.normal(profit, 2, 1000)
        
        # 绘制直方图
        ax.hist(profits, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加垂直线标记期望利润
        ax.axvline(profit, color='red', linestyle='--', linewidth=3, label=f'期望利润: {profit:.2f}')
        
        # 添加统计信息
        ax.text(0.05, 0.95, f'均值: {np.mean(profits):.2f}\n标准差: {np.std(profits):.2f}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 添加决策方案说明
        decision_text = []
        if decisions.get('test_component1', False):
            decision_text.append('检测零件1')
        if decisions.get('test_component2', False):
            decision_text.append('检测零件2')
        if decisions.get('test_finished_product', False):
            decision_text.append('检测成品')
        if decisions.get('disassemble_defective', False):
            decision_text.append('拆解不合格品')
        
        if not decision_text:
            decision_text = ['无检测措施']
        
        decision_str = '\n'.join(decision_text)
        ax.text(0.05, 0.7, f'决策方案:\n{decision_str}', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        ax.set_xlabel('利润 (元)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.set_title(f'情况{case_id}利润分布', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _create_network_plot(self, G: nx.DiGraph, decisions: Dict, cost_analysis: Dict = None):
        """创建专业网络可视化"""
        fig = plt.figure(figsize=(22, 18))
        gs = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.8, height_ratios=[3, 1])
        
        # 主网络图
        ax_main = fig.add_subplot(gs[0, :])
        
        # 使用分层布局
        pos = {}
        layers = {
            'components': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
            'semi_products': ['SP1', 'SP2', 'SP3'],
            'final_product': ['FP']
        }
        
        # 设置分层位置
        y_positions = {'components': 0, 'semi_products': 1, 'final_product': 2}
        for layer_name, nodes in layers.items():
            y = y_positions[layer_name]
            for i, node in enumerate(nodes):
                if node in G.nodes():
                    x = (i - len(nodes)/2) * 2
                    pos[node] = (x, y)
        
        # 绘制节点 - 根据类型和决策状态着色
        node_colors = []
        node_sizes = []
        edge_colors = []
        
        for node in G.nodes():
            if node.startswith('C'):  # 零件
                if decisions.get(node, False):
                    node_colors.append('#2E8B57')  # 深绿色 - 检测
                    node_sizes.append(1200)
                else:
                    node_colors.append('#FF6B6B')  # 红色 - 不检测
                    node_sizes.append(800)
            elif node.startswith('SP'):  # 半成品
                if decisions.get(node, False):
                    node_colors.append('#4169E1')  # 蓝色 - 检测
                    node_sizes.append(1500)
                else:
                    node_colors.append('#FFA500')  # 橙色 - 不检测
                    node_sizes.append(1000)
            else:  # 成品
                if decisions.get(node, False):
                    node_colors.append('#8A2BE2')  # 紫色 - 检测
                    node_sizes.append(2000)
                else:
                    node_colors.append('#DC143C')  # 深红色 - 不检测
                    node_sizes.append(1500)
        
        # 绘制边
        for edge in G.edges():
            if decisions.get(edge[0], False) and decisions.get(edge[1], False):
                edge_colors.append('#00FF00')  # 绿色 - 两端都检测
            elif decisions.get(edge[0], False) or decisions.get(edge[1], False):
                edge_colors.append('#FFFF00')  # 黄色 - 一端检测
            else:
                edge_colors.append('#FF0000')  # 红色 - 都不检测
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8, ax=ax_main)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                              arrows=True, arrowsize=30, width=3, 
                              alpha=0.7, ax=ax_main)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax_main)
        
        ax_main.set_title('多工序生产网络优化决策图', fontsize=18, fontweight='bold', pad=20)
        ax_main.axis('off')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E8B57', 
                      markersize=15, label='零件(检测)', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                      markersize=12, label='零件(不检测)', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', 
                      markersize=17, label='半成品(检测)', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', 
                      markersize=14, label='半成品(不检测)', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8A2BE2', 
                      markersize=20, label='成品(检测)', linestyle='None'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C', 
                      markersize=17, label='成品(不检测)', linestyle='None')
        ]
        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=11, bbox_to_anchor=(1.0, 1.0))
        
        # 决策统计图
        ax_stats = fig.add_subplot(gs[1, 0])
        test_count = sum(1 for decision in decisions.values() if decision)
        no_test_count = len(decisions) - test_count
        
        labels = ['检测', '不检测']
        sizes = [test_count, no_test_count]
        colors = ['#32CD32', '#FF4500']
        explode = (0.1, 0)
        
        # 创建饼图，完全不显示任何标签，无阴影效果
        wedges, texts = ax_stats.pie(sizes, explode=explode, labels=None, colors=colors,
                                     autopct=None, shadow=False, startangle=90)
        
        # 专业的极简标签设计 - 完全避免与饼图重叠
        # 将标签放在左侧更远的位置，使用垂直布局
        test_pct = test_count/(test_count+no_test_count)*100
        no_test_pct = no_test_count/(test_count+no_test_count)*100
        
        # 检测标签组 - 左上方
        ax_stats.text(-2.8, 0.6, '✅ 检测决策', fontsize=12, fontweight='bold', ha='left', va='center',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#32CD32', alpha=0.95, edgecolor='darkgreen'))
        ax_stats.text(-2.8, 0.35, f'比例: {test_pct:.1f}%', fontsize=11, ha='left', va='center')
        ax_stats.text(-2.8, 0.15, f'数量: {test_count}个', fontsize=10, ha='left', va='center', color='gray')
        
        # 不检测标签组 - 左下方  
        ax_stats.text(-2.8, -0.15, '❌ 不检测决策', fontsize=12, fontweight='bold', ha='left', va='center',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#FF4500', alpha=0.95, edgecolor='darkred'))
        ax_stats.text(-2.8, -0.4, f'比例: {no_test_pct:.1f}%', fontsize=11, ha='left', va='center')
        ax_stats.text(-2.8, -0.6, f'数量: {no_test_count}个', fontsize=10, ha='left', va='center', color='gray')
        
        # 扩大显示区域，确保左侧标签完全可见
        ax_stats.set_xlim(-3.5, 1.5)
        ax_stats.set_ylim(-1, 1)
        
        ax_stats.set_title('检测决策分布', fontsize=14, fontweight='bold', pad=15)
        
        # 成本效益分析表
        ax_table = fig.add_subplot(gs[1, 1])
        ax_table.axis('off')
        
        if cost_analysis:
            table_data = [
                ['项目', '数值', '单位'],
                ['总检测成本', f"{cost_analysis.get('total_test_cost', 0):.2f}", '元'],
                ['预期缺陷损失', f"{cost_analysis.get('defect_loss', 0):.2f}", '元'],
                ['总成本', f"{cost_analysis.get('total_cost', 0):.2f}", '元'],
                ['检测节点数', f"{test_count}", '个'],
                ['优化效率', f"{cost_analysis.get('efficiency', 0):.1f}", '%']
            ]
        else:
            table_data = [
                ['项目', '数值', '单位'],
                ['检测节点数', f"{test_count}", '个'],
                ['非检测节点数', f"{no_test_count}", '个'],
                ['总节点数', f"{len(decisions)}", '个'],
                ['检测比例', f"{test_count/len(decisions)*100:.1f}", '%']
            ]
        
        table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0], 
                              cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.5)
        
        # 设置表格样式
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax_table.set_title('优化结果汇总', fontsize=14, fontweight='bold', pad=15)
        
        plt.suptitle('多工序生产网络专业分析报告', fontsize=20, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.output_dir / "problem3_multistage_network.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_uncertainty_analysis_plot(self, uncertainty_ranges: Dict, robust_results: Dict = None):
        """创建专业不确定性分析图"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 主要参数不确定性分布（前6个参数）
        param_items = list(uncertainty_ranges.items())[:6]
        for i, (param, (min_val, max_val)) in enumerate(param_items):
            row, col = i // 3, i % 3
            ax = fig.add_subplot(gs[row, col])
            
            # 生成多种分布样本进行对比
            uniform_samples = np.random.uniform(min_val, max_val, 1000)
            normal_samples = np.random.normal((min_val + max_val)/2, (max_val - min_val)/6, 1000)
            normal_samples = np.clip(normal_samples, min_val, max_val)
            
            # 绘制分布对比
            ax.hist(uniform_samples, bins=30, alpha=0.6, color='skyblue', 
                   label='均匀分布', density=True, edgecolor='navy')
            ax.hist(normal_samples, bins=30, alpha=0.6, color='lightcoral', 
                   label='正态分布', density=True, edgecolor='darkred')
            
            # 标记关键点
            ax.axvline(min_val, color='red', linestyle='--', linewidth=2, label='下界')
            ax.axvline(max_val, color='red', linestyle='--', linewidth=2, label='上界')
            ax.axvline((min_val + max_val)/2, color='green', linestyle=':', linewidth=2, label='中值')
            
            # 填充不确定性区间
            ax.fill_between([min_val, max_val], 0, ax.get_ylim()[1], 
                           alpha=0.2, color='yellow', label='不确定区间')
            
            ax.set_title(f'{param}\n不确定性范围: [{min_val:.3f}, {max_val:.3f}]', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('参数值', fontsize=10)
            ax.set_ylabel('概率密度', fontsize=10)
            if i == 0:  # 只在第一个子图显示图例
                ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # 蒙特卡洛模拟结果分析
        ax_monte = fig.add_subplot(gs[2, :])
        
        if robust_results and 'monte_carlo_costs' in robust_results:
            costs = robust_results['monte_carlo_costs']
            
            # 绘制成本分布
            n, bins, patches = ax_monte.hist(costs, bins=50, alpha=0.7, color='lightgreen', 
                                           edgecolor='darkgreen', density=True)
            
            # 计算统计量
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            percentiles = np.percentile(costs, [5, 25, 50, 75, 95])
            
            # 标记统计量
            ax_monte.axvline(mean_cost, color='red', linestyle='-', linewidth=3, label=f'均值: {mean_cost:.2f}')
            ax_monte.axvline(percentiles[2], color='blue', linestyle='--', linewidth=2, label=f'中位数: {percentiles[2]:.2f}')
            ax_monte.axvspan(percentiles[1], percentiles[3], alpha=0.3, color='blue', label='25%-75%区间')
            ax_monte.axvspan(percentiles[0], percentiles[4], alpha=0.2, color='orange', label='5%-95%区间')
            
            # 添加正态分布拟合曲线
            x = np.linspace(min(costs), max(costs), 100)
            normal_curve = norm.pdf(x, mean_cost, std_cost)
            ax_monte.plot(x, normal_curve, 'r-', linewidth=2, label='正态拟合')
            
            ax_monte.set_title('蒙特卡洛模拟 - 鲁棒成本分布分析', fontsize=14, fontweight='bold')
            ax_monte.set_xlabel('总成本', fontsize=12)
            ax_monte.set_ylabel('概率密度', fontsize=12)
            ax_monte.legend(fontsize=10)
            ax_monte.grid(True, alpha=0.3)
            
            # 添加统计信息文本框
            stats_text = f'统计信息:\n均值: {mean_cost:.2f}\n标准差: {std_cost:.2f}\n变异系数: {std_cost/mean_cost:.3f}\n5%分位数: {percentiles[0]:.2f}\n95%分位数: {percentiles[4]:.2f}'
            ax_monte.text(0.02, 0.98, stats_text, transform=ax_monte.transAxes, 
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                         fontsize=10)
        else:
            # 如果没有蒙特卡洛结果，显示不确定性影响分析
            ax_monte.text(0.5, 0.5, '鲁棒优化不确定性影响分析\n（需要蒙特卡洛模拟结果）', 
                         transform=ax_monte.transAxes, ha='center', va='center',
                         fontsize=16, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax_monte.set_title('鲁棒优化不确定性影响分析', fontsize=14, fontweight='bold')
        
        plt.suptitle('鲁棒优化不确定性专业分析报告', fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(self.output_dir / "problem4_uncertainty_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def solve_all_problems(self) -> Dict:
        """求解所有问题"""
        logger.info("开始求解2024年数学建模竞赛B题...")
        
        results = {}
        
        # 问题1：抽样检验
        try:
            results['problem1'] = self.solve_problem1_sampling()
            logger.info("问题1求解完成")
        except Exception as e:
            logger.error(f"问题1求解失败: {e}")
            results['problem1'] = {'error': str(e)}
        
        # 问题2：生产决策（所有6种情况）
        results['problem2'] = {}
        for case_id in range(1, 7):
            try:
                results['problem2'][f'case_{case_id}'] = self.solve_problem2_production(case_id)
                logger.info(f"问题2情况{case_id}求解完成")
            except Exception as e:
                logger.error(f"问题2情况{case_id}求解失败: {e}")
                results['problem2'][f'case_{case_id}'] = {'error': str(e)}
        
        # 问题3：多工序优化
        try:
            results['problem3'] = self.solve_problem3_multistage()
            logger.info("问题3求解完成")
        except Exception as e:
            logger.error(f"问题3求解失败: {e}")
            results['problem3'] = {'error': str(e)}
        
        # 问题4：鲁棒优化
        try:
            results['problem4'] = self.solve_problem4_uncertainty()
            logger.info("问题4求解完成")
        except Exception as e:
            logger.error(f"问题4求解失败: {e}")
            results['problem4'] = {'error': str(e)}
        
        # 生成综合报告
        self._generate_comprehensive_report(results)
        
        logger.info("所有问题求解完成！")
        return results

    def _generate_comprehensive_report(self, results: Dict):
        """生成综合报告"""
        report_path = self.output_dir / "competition_b_comprehensive_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("2024年高教社杯全国大学生数学建模竞赛B题求解报告\n")
            f.write("=" * 60 + "\n\n")
            
            # 问题1报告
            f.write("问题1：供应商批次验收决策\n")
            f.write("-" * 30 + "\n")
            if 'error' not in results['problem1']:
                p1 = results['problem1']
                f.write(f"最优样本量: {p1['sample_size']}\n")
                f.write(f"判定值: {p1['acceptance_number']}\n")
                f.write(f"实际第一类错误概率: {p1['actual_alpha']:.4f}\n")
                f.write(f"实际第二类错误概率: {p1['actual_beta']:.4f}\n")
                f.write(f"可视化图表: {p1['plot_path']}\n")
            else:
                f.write(f"求解失败: {results['problem1']['error']}\n")
            f.write("\n")
            
            # 问题2报告
            f.write("问题2：生产流程阶段决策\n")
            f.write("-" * 30 + "\n")
            for case_id in range(1, 7):
                case_key = f'case_{case_id}'
                if case_key in results['problem2']:
                    case_result = results['problem2'][case_key]
                    f.write(f"情况{case_id}:\n")
                    if 'error' not in case_result:
                        decisions = case_result['decisions']
                        f.write(f"  检测零件1: {'是' if decisions['test_component1'] else '否'}\n")
                        f.write(f"  检测零件2: {'是' if decisions['test_component2'] else '否'}\n")
                        f.write(f"  检测成品: {'是' if decisions['test_finished_product'] else '否'}\n")
                        f.write(f"  拆解不合格品: {'是' if decisions['disassemble_defective'] else '否'}\n")
                        f.write(f"  期望利润: {case_result['expected_profit']:.2f}\n")
                        f.write(f"  可视化图表: {case_result['plot_path']}\n")
                    else:
                        f.write(f"  求解失败: {case_result['error']}\n")
            f.write("\n")
            
            # 问题3报告
            f.write("问题3：多工序网络优化\n")
            f.write("-" * 30 + "\n")
            if 'error' not in results['problem3']:
                p3 = results['problem3']
                f.write(f"网络规模: {p3['network_size']}个节点\n")
                f.write(f"总成本: {p3['total_cost']:.2f}\n")
                f.write(f"可视化图表: {p3['plot_path']}\n")
            else:
                f.write(f"求解失败: {results['problem3']['error']}\n")
            f.write("\n")
            
            # 问题4报告
            f.write("问题4：鲁棒优化分析\n")
            f.write("-" * 30 + "\n")
            if 'error' not in results['problem4']:
                p4 = results['problem4']
                f.write(f"最坏情况利润: {p4['worst_case_profit']:.2f}\n")
                f.write(f"可视化图表: {p4['plot_path']}\n")
            else:
                f.write(f"求解失败: {results['problem4']['error']}\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("报告生成完成！\n")
        
        logger.info(f"综合报告已生成: {report_path}")

def run_competition_b_solver():
    """运行竞赛B题求解器"""
    solver = CompetitionBSolver()
    results = solver.solve_all_problems()
    return results

if __name__ == "__main__":
    run_competition_b_solver() 