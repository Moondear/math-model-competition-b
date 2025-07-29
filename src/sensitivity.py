"""
敏感性分析可视化模块
用于分析模型参数变化对结果的影响
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple
import logging
import os
from .font_utils import setup_chinese_font, ensure_output_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """敏感性分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.results = {}
        
    def analyze_parameter_sensitivity(self, 
                                   base_params: Dict,
                                   param_ranges: Dict[str, Tuple[float, float]],
                                   n_samples: int = 1000) -> Dict:
        """分析参数敏感性
        
        Args:
            base_params: 基准参数
            param_ranges: 参数变化范围 {参数名: (最小值, 最大值)}
            n_samples: 采样数量
            
        Returns:
            Dict: 敏感性分析结果
        """
        logger.info("开始参数敏感性分析...")
        
        # 生成参数样本
        samples = {}
        for param, (min_val, max_val) in param_ranges.items():
            samples[param] = np.linspace(min_val, max_val, n_samples)
        
        # 计算目标函数值
        results = {}
        for param, values in samples.items():
            param_results = []
            for value in values:
                # 创建测试参数
                test_params = base_params.copy()
                test_params[param] = value
                
                # 计算目标函数（这里用简化的利润函数）
                profit = self._calculate_profit(test_params)
                param_results.append(profit)
            
            results[param] = {
                'values': values,
                'profits': param_results,
                'sensitivity': np.std(param_results) / np.mean(param_results)
            }
        
        self.results = results
        return results
    
    def _calculate_profit(self, params: Dict) -> float:
        """计算利润函数
        
        Args:
            params: 参数字典
            
        Returns:
            float: 期望利润
        """
        # 简化的利润计算模型
        defect_rate1 = params.get('defect_rate1', 0.1)
        defect_rate2 = params.get('defect_rate2', 0.1)
        test_cost1 = params.get('test_cost1', 2)
        test_cost2 = params.get('test_cost2', 3)
        assembly_cost = params.get('assembly_cost', 6)
        market_price = params.get('market_price', 56)
        
        # 计算合格率
        p_ok = (1 - defect_rate1) * (1 - defect_rate2)
        
        # 计算总成本
        total_cost = test_cost1 + test_cost2 + assembly_cost
        
        # 计算期望利润
        expected_profit = p_ok * market_price - total_cost
        
        return expected_profit
    
    def monte_carlo_simulation(self, 
                             base_params: Dict,
                             param_distributions: Dict[str, Tuple[str, float, float]],
                             n_simulations: int = 10000) -> Dict:
        """蒙特卡罗模拟
        
        Args:
            base_params: 基准参数
            param_distributions: 参数分布 {参数名: (分布类型, 参数1, 参数2)}
            n_simulations: 模拟次数
            
        Returns:
            Dict: 模拟结果
        """
        logger.info("开始蒙特卡罗模拟...")
        
        # 生成随机参数
        random_params = []
        for _ in range(n_simulations):
            param_set = base_params.copy()
            for param, (dist_type, param1, param2) in param_distributions.items():
                if dist_type == 'normal':
                    value = np.random.normal(param1, param2)
                elif dist_type == 'uniform':
                    value = np.random.uniform(param1, param2)
                else:
                    value = param1  # 默认值
                
                # 确保参数在合理范围内
                if 'rate' in param:
                    value = np.clip(value, 0, 1)
                elif 'cost' in param or 'price' in param:
                    value = np.clip(value, 0, 1000)
                
                param_set[param] = value
            
            random_params.append(param_set)
        
        # 计算利润分布
        profits = [self._calculate_profit(params) for params in random_params]
        
        # 统计分析
        profit_mean = np.mean(profits)
        profit_std = np.std(profits)
        profit_95ci = np.percentile(profits, [2.5, 97.5])
        
        return {
            'profits': profits,
            'mean': profit_mean,
            'std': profit_std,
            'ci_95': profit_95ci,
            'min': np.min(profits),
            'max': np.max(profits)
        }
    
    def create_sensitivity_plots(self, save_dir: str = "output") -> List[str]:
        """创建敏感性分析图表
        
        Args:
            save_dir: 保存目录
            
        Returns:
            List[str]: 图表文件路径列表
        """
        # 设置中文字体
        setup_chinese_font()
        ensure_output_dir()
        
        if not self.results:
            logger.warning("没有敏感性分析结果，请先运行分析")
            return []
        
        saved_files = []
        
        # 1. 参数敏感性曲线图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (param, result) in enumerate(self.results.items()):
            if i < 4:  # 只显示前4个参数
                ax = axes[i]
                ax.plot(result['values'], result['profits'], 'b-', linewidth=2)
                ax.set_xlabel(param)
                ax.set_ylabel('期望利润')
                ax.set_title(f'{param}敏感性分析')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        sensitivity_curve_path = f"{save_dir}/sensitivity_curves.png"
        plt.savefig(sensitivity_curve_path, dpi=300, bbox_inches='tight')
        saved_files.append(sensitivity_curve_path)
        plt.close()
        
        # 2. 敏感性热力图
        sensitivity_scores = {param: result['sensitivity'] 
                            for param, result in self.results.items()}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        params = list(sensitivity_scores.keys())
        scores = list(sensitivity_scores.values())
        
        bars = ax.bar(params, scores, color='skyblue', alpha=0.7)
        ax.set_xlabel('参数')
        ax.set_ylabel('敏感性指数')
        ax.set_title('参数敏感性排序')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        sensitivity_heatmap_path = f"{save_dir}/sensitivity_heatmap.png"
        plt.savefig(sensitivity_heatmap_path, dpi=300, bbox_inches='tight')
        saved_files.append(sensitivity_heatmap_path)
        plt.close()
        
        # 3. 交互式3D敏感性图
        if len(self.results) >= 2:
            param1, param2 = list(self.results.keys())[:2]
            x = self.results[param1]['values']
            y = self.results[param2]['values']
            X, Y = np.meshgrid(x, y)
            
            # 计算Z值（简化计算）
            Z = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    test_params = {
                        param1: x[i],
                        param2: y[j]
                    }
                    Z[j, i] = self._calculate_profit(test_params)
            
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
            fig.update_layout(
                title=f'{param1} vs {param2} 敏感性分析',
                scene=dict(
                    xaxis_title=param1,
                    yaxis_title=param2,
                    zaxis_title='期望利润'
                )
            )
            
            sensitivity_3d_path = f"{save_dir}/sensitivity_3d.html"
            fig.write_html(sensitivity_3d_path)
            saved_files.append(sensitivity_3d_path)
        
        return saved_files
    
    def create_monte_carlo_plots(self, mc_results: Dict, save_dir: str = "output") -> List[str]:
        """创建蒙特卡罗模拟图表
        
        Args:
            mc_results: 蒙特卡罗模拟结果
            save_dir: 保存目录
            
        Returns:
            List[str]: 图表文件路径列表
        """
        # 设置中文字体
        setup_chinese_font()
        ensure_output_dir()
        
        saved_files = []
        
        # 1. 利润分布直方图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.hist(mc_results['profits'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(mc_results['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {mc_results["mean"]:.2f}')
        ax.axvline(mc_results['ci_95'][0], color='orange', linestyle=':', linewidth=2,
                   label=f'95%置信区间: [{mc_results["ci_95"][0]:.2f}, {mc_results["ci_95"][1]:.2f}]')
        ax.axvline(mc_results['ci_95'][1], color='orange', linestyle=':', linewidth=2)
        
        ax.set_xlabel('期望利润')
        ax.set_ylabel('频次')
        ax.set_title('蒙特卡罗模拟 - 利润分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mc_hist_path = f"{save_dir}/monte_carlo_histogram.png"
        plt.savefig(mc_hist_path, dpi=300, bbox_inches='tight')
        saved_files.append(mc_hist_path)
        plt.close()
        
        # 2. 累积分布函数
        profits_sorted = np.sort(mc_results['profits'])
        cdf = np.arange(1, len(profits_sorted) + 1) / len(profits_sorted)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(profits_sorted, cdf, 'b-', linewidth=2)
        ax.axvline(mc_results['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'均值: {mc_results["mean"]:.2f}')
        ax.axvline(mc_results['ci_95'][0], color='orange', linestyle=':', linewidth=2,
                   label=f'95%置信区间')
        ax.axvline(mc_results['ci_95'][1], color='orange', linestyle=':', linewidth=2)
        
        ax.set_xlabel('期望利润')
        ax.set_ylabel('累积概率')
        ax.set_title('蒙特卡罗模拟 - 累积分布函数')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        mc_cdf_path = f"{save_dir}/monte_carlo_cdf.png"
        plt.savefig(mc_cdf_path, dpi=300, bbox_inches='tight')
        saved_files.append(mc_cdf_path)
        plt.close()
        
        # 3. 交互式箱线图
        fig = go.Figure()
        fig.add_trace(go.Box(y=mc_results['profits'], name='利润分布'))
        fig.update_layout(
            title='蒙特卡罗模拟 - 利润分布箱线图',
            yaxis_title='期望利润',
            showlegend=False
        )
        
        mc_box_path = f"{save_dir}/monte_carlo_boxplot.html"
        fig.write_html(mc_box_path)
        saved_files.append(mc_box_path)
        
        return saved_files

def run_sensitivity_analysis():
    """运行敏感性分析演示"""
    print("🔍 开始敏感性分析...")
    
    # 创建分析器
    analyzer = SensitivityAnalyzer()
    
    # 基准参数
    base_params = {
        'defect_rate1': 0.1,
        'defect_rate2': 0.1,
        'test_cost1': 2,
        'test_cost2': 3,
        'assembly_cost': 6,
        'market_price': 56
    }
    
    # 参数变化范围
    param_ranges = {
        'defect_rate1': (0.05, 0.25),
        'defect_rate2': (0.05, 0.25),
        'test_cost1': (1, 5),
        'test_cost2': (2, 8),
        'assembly_cost': (4, 10),
        'market_price': (40, 80)
    }
    
    # 运行敏感性分析
    sensitivity_results = analyzer.analyze_parameter_sensitivity(base_params, param_ranges)
    
    print("\n📊 敏感性分析结果:")
    for param, result in sensitivity_results.items():
        print(f"  {param}: 敏感性指数 = {result['sensitivity']:.4f}")
    
    # 参数分布设置
    param_distributions = {
        'defect_rate1': ('normal', 0.1, 0.02),
        'defect_rate2': ('normal', 0.1, 0.02),
        'test_cost1': ('uniform', 1.5, 2.5),
        'test_cost2': ('uniform', 2.5, 3.5),
        'assembly_cost': ('normal', 6, 0.5),
        'market_price': ('normal', 56, 5)
    }
    
    # 运行蒙特卡罗模拟
    mc_results = analyzer.monte_carlo_simulation(base_params, param_distributions)
    
    print(f"\n📈 蒙特卡罗模拟结果:")
    print(f"  平均利润: {mc_results['mean']:.2f}")
    print(f"  标准差: {mc_results['std']:.2f}")
    print(f"  95%置信区间: [{mc_results['ci_95'][0]:.2f}, {mc_results['ci_95'][1]:.2f}]")
    print(f"  最小值: {mc_results['min']:.2f}")
    print(f"  最大值: {mc_results['max']:.2f}")
    
    # 生成图表
    print("\n🎨 正在生成可视化图表...")
    sensitivity_plots = analyzer.create_sensitivity_plots()
    mc_plots = analyzer.create_monte_carlo_plots(mc_results)
    
    all_plots = sensitivity_plots + mc_plots
    print(f"✅ 已生成 {len(all_plots)} 个图表文件:")
    for plot_path in all_plots:
        print(f"  📊 {plot_path}")
    
    return {
        'sensitivity_results': sensitivity_results,
        'mc_results': mc_results,
        'plot_files': all_plots
    }

if __name__ == "__main__":
    run_sensitivity_analysis() 