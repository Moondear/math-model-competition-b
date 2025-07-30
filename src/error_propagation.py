"""
误差传播模型
实现抽样误差的数学建模和传播分析
"""

import numpy as np
from scipy import stats
from scipy.stats import beta, binom, norm
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
from .font_utils import setup_chinese_font, ensure_output_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ErrorPropagationParams:
    """误差传播参数"""
    confidence_level: float = 0.95  # 置信水平
    sample_size: int = 100          # 样本量
    prior_alpha: float = 1.0        # Beta分布先验参数α
    prior_beta: float = 1.0         # Beta分布先验参数β
    monte_carlo_iterations: int = 10000  # 蒙特卡罗模拟次数

class SamplingErrorModel:
    """抽样误差传播模型"""
    
    def __init__(self, params: ErrorPropagationParams):
        """初始化误差模型
        
        Args:
            params: 误差传播参数
        """
        self.params = params
        self.posterior_distributions = {}
        
    def apply_bayesian_update(self, defect_rate: float, 
                            observed_defects: int, 
                            sample_size: int) -> Dict:
        """应用贝叶斯更新修正次品率估计
        
        Args:
            defect_rate: 原始次品率估计
            observed_defects: 观察到的次品数
            sample_size: 样本量
            
        Returns:
            Dict: 贝叶斯更新结果
        """
        logger.info(f"应用贝叶斯更新: 次品率={defect_rate}, 次品数={observed_defects}, 样本量={sample_size}")
        
        # 使用共轭先验（Beta分布）
        # 先验: Beta(α, β)
        # 似然: Binomial(n, p)
        # 后验: Beta(α + x, β + n - x)
        
        prior_alpha = self.params.prior_alpha
        prior_beta = self.params.prior_beta
        
        # 贝叶斯更新
        posterior_alpha = prior_alpha + observed_defects
        posterior_beta = prior_beta + sample_size - observed_defects
        
        # 后验分布
        posterior_dist = beta(posterior_alpha, posterior_beta)
        
        # 点估计（后验均值）
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        
        # 置信区间
        alpha_level = 1 - self.params.confidence_level
        lower_bound = posterior_dist.ppf(alpha_level / 2)
        upper_bound = posterior_dist.ppf(1 - alpha_level / 2)
        
        # 计算不确定性度量
        posterior_var = posterior_dist.var()
        credible_interval_width = upper_bound - lower_bound
        
        return {
            'original_estimate': defect_rate,
            'posterior_mean': posterior_mean,
            'posterior_variance': posterior_var,
            'credible_interval': (lower_bound, upper_bound),
            'credible_interval_width': credible_interval_width,
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'uncertainty_reduction': abs(defect_rate - posterior_mean) / defect_rate if defect_rate > 0 else 0
        }
    
    def calculate_confidence_interval(self, defect_rate: float, 
                                    sample_size: int) -> Tuple[float, float]:
        """计算次品率的置信区间
        
        Args:
            defect_rate: 次品率点估计
            sample_size: 样本量
            
        Returns:
            Tuple[float, float]: 置信区间的下界和上界
        """
        if defect_rate <= 0 or defect_rate >= 1:
            raise ValueError("次品率必须在(0,1)区间内")
        
        # 使用Clopper-Pearson精确方法
        observed_defects = int(defect_rate * sample_size)
        alpha = 1 - self.params.confidence_level
        
        # 下界：Beta分布的分位数
        if observed_defects == 0:
            lower = 0
        else:
            lower = beta.ppf(alpha/2, observed_defects, sample_size - observed_defects + 1)
        
        # 上界：Beta分布的分位数
        if observed_defects == sample_size:
            upper = 1
        else:
            upper = beta.ppf(1 - alpha/2, observed_defects + 1, sample_size - observed_defects)
        
        return (max(0, lower), min(1, upper))
    
    def propagate_uncertainty(self, input_uncertainties: List[Dict], 
                            propagation_function) -> Dict:
        """使用蒙特卡罗方法传播不确定性
        
        Args:
            input_uncertainties: 输入变量的不确定性描述列表
            propagation_function: 传播函数，接受输入变量列表，返回输出值
            
        Returns:
            Dict: 输出不确定性分析结果
        """
        logger.info(f"开始不确定性传播分析，{self.params.monte_carlo_iterations}次蒙特卡罗模拟")
        
        output_samples = []
        
        for _ in range(self.params.monte_carlo_iterations):
            # 从每个输入变量的分布中采样
            input_samples = []
            for uncertainty in input_uncertainties:
                if uncertainty['distribution'] == 'beta':
                    sample = beta.rvs(uncertainty['alpha'], uncertainty['beta'])
                elif uncertainty['distribution'] == 'normal':
                    sample = norm.rvs(uncertainty['mean'], uncertainty['std'])
                elif uncertainty['distribution'] == 'uniform':
                    sample = np.random.uniform(uncertainty['min'], uncertainty['max'])
                else:
                    raise ValueError(f"不支持的分布类型: {uncertainty['distribution']}")
                
                input_samples.append(sample)
            
            # 应用传播函数
            try:
                output = propagation_function(input_samples)
                output_samples.append(output)
            except Exception as e:
                logger.warning(f"传播函数执行失败: {e}")
                continue
        
        if not output_samples:
            raise ValueError("所有蒙特卡罗样本都失败了")
        
        output_samples = np.array(output_samples)
        
        # 统计分析
        mean_output = np.mean(output_samples)
        std_output = np.std(output_samples)
        
        # 置信区间
        alpha = 1 - self.params.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        confidence_interval = (
            np.percentile(output_samples, lower_percentile),
            np.percentile(output_samples, upper_percentile)
        )
        
        return {
            'mean': mean_output,
            'std': std_output,
            'variance': std_output**2,
            'confidence_interval': confidence_interval,
            'samples': output_samples,
            'sample_count': len(output_samples),
            'coefficient_of_variation': std_output / abs(mean_output) if mean_output != 0 else np.inf
        }
    
    def sensitivity_analysis(self, base_values: List[float], 
                           perturbation_size: float,
                           evaluation_function) -> Dict:
        """敏感性分析：分析输入变量对输出的影响
        
        Args:
            base_values: 基准输入值列表
            perturbation_size: 扰动大小（相对值）
            evaluation_function: 评估函数
            
        Returns:
            Dict: 敏感性分析结果
        """
        logger.info(f"开始敏感性分析，扰动大小={perturbation_size}")
        
        base_output = evaluation_function(base_values)
        sensitivities = {}
        
        for i, base_value in enumerate(base_values):
            # 正向扰动
            perturbed_values_pos = base_values.copy()
            perturbed_values_pos[i] = base_value * (1 + perturbation_size)
            output_pos = evaluation_function(perturbed_values_pos)
            
            # 负向扰动
            perturbed_values_neg = base_values.copy()
            perturbed_values_neg[i] = base_value * (1 - perturbation_size)
            output_neg = evaluation_function(perturbed_values_neg)
            
            # 计算敏感性（数值导数）
            sensitivity = (output_pos - output_neg) / (2 * base_value * perturbation_size)
            relative_sensitivity = sensitivity * base_value / base_output if base_output != 0 else 0
            
            sensitivities[f'variable_{i}'] = {
                'absolute_sensitivity': sensitivity,
                'relative_sensitivity': relative_sensitivity,
                'base_value': base_value,
                'output_change_pos': output_pos - base_output,
                'output_change_neg': output_neg - base_output
            }
        
        return {
            'base_output': base_output,
            'sensitivities': sensitivities,
            'most_sensitive_variable': max(sensitivities.keys(), 
                                         key=lambda k: abs(sensitivities[k]['relative_sensitivity'])),
            'sensitivity_ranking': sorted(sensitivities.keys(), 
                                        key=lambda k: abs(sensitivities[k]['relative_sensitivity']), 
                                        reverse=True)
        }
    
    def create_error_propagation_report(self, analysis_results: Dict, 
                                      output_path: str = None) -> str:
        """生成误差传播分析报告
        
        Args:
            analysis_results: 分析结果字典
            output_path: 输出路径
            
        Returns:
            str: 报告文件路径
        """
        ensure_output_dir()
        
        if output_path is None:
            output_path = "output/error_propagation_report.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("误差传播分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            if 'bayesian_update' in analysis_results:
                f.write("1. 贝叶斯更新结果\n")
                f.write("-" * 30 + "\n")
                bu = analysis_results['bayesian_update']
                f.write(f"原始估计: {bu['original_estimate']:.4f}\n")
                f.write(f"后验均值: {bu['posterior_mean']:.4f}\n")
                f.write(f"后验方差: {bu['posterior_variance']:.6f}\n")
                f.write(f"可信区间: [{bu['credible_interval'][0]:.4f}, {bu['credible_interval'][1]:.4f}]\n")
                f.write(f"不确定性减少: {bu['uncertainty_reduction']:.2%}\n\n")
            
            if 'uncertainty_propagation' in analysis_results:
                f.write("2. 不确定性传播结果\n")
                f.write("-" * 30 + "\n")
                up = analysis_results['uncertainty_propagation']
                f.write(f"输出均值: {up['mean']:.4f}\n")
                f.write(f"输出标准差: {up['std']:.4f}\n")
                f.write(f"置信区间: [{up['confidence_interval'][0]:.4f}, {up['confidence_interval'][1]:.4f}]\n")
                f.write(f"变异系数: {up['coefficient_of_variation']:.4f}\n\n")
            
            if 'sensitivity_analysis' in analysis_results:
                f.write("3. 敏感性分析结果\n")
                f.write("-" * 30 + "\n")
                sa = analysis_results['sensitivity_analysis']
                f.write(f"基准输出: {sa['base_output']:.4f}\n")
                f.write(f"最敏感变量: {sa['most_sensitive_variable']}\n")
                f.write("敏感性排序:\n")
                for var in sa['sensitivity_ranking']:
                    sens = sa['sensitivities'][var]
                    f.write(f"  {var}: 相对敏感性 = {sens['relative_sensitivity']:.4f}\n")
        
        logger.info(f"误差传播报告已保存至: {output_path}")
        return output_path
    
    def visualize_uncertainty(self, samples: np.ndarray, 
                            title: str = "不确定性分布",
                            output_path: str = None) -> str:
        """可视化不确定性分布
        
        Args:
            samples: 蒙特卡罗样本
            title: 图表标题
            output_path: 输出路径
            
        Returns:
            str: 图表文件路径
        """
        ensure_output_dir()
        setup_chinese_font()
        
        if output_path is None:
            output_path = "output/uncertainty_distribution.png"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 直方图
        ax1.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('输出值')
        ax1.set_ylabel('概率密度')
        ax1.set_title(f'{title} - 直方图')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q图检验正态性
        stats.probplot(samples, dist="norm", plot=ax2)
        ax2.set_title(f'{title} - Q-Q图')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"不确定性分布图已保存至: {output_path}")
        return output_path


def production_profit_function(variables: List[float]) -> float:
    """生产利润函数示例
    
    Args:
        variables: [defect_rate1, defect_rate2, market_price, test_cost1, test_cost2]
        
    Returns:
        float: 期望利润
    """
    defect_rate1, defect_rate2, market_price, test_cost1, test_cost2 = variables
    
    # 简化的利润计算模型
    good_rate = (1 - defect_rate1) * (1 - defect_rate2)
    total_cost = test_cost1 + test_cost2 + 6  # 装配成本
    
    # 考虑次品处理成本
    defect_cost = (defect_rate1 + defect_rate2) * 10  # 次品处理成本
    
    profit = good_rate * market_price - total_cost - defect_cost
    return max(0, profit)  # 利润不能为负


if __name__ == "__main__":
    # 测试误差传播模型
    params = ErrorPropagationParams(confidence_level=0.95, monte_carlo_iterations=5000)
    model = SamplingErrorModel(params)
    
    # 测试贝叶斯更新
    bayesian_result = model.apply_bayesian_update(
        defect_rate=0.1, observed_defects=8, sample_size=100
    )
    
    # 测试不确定性传播
    input_uncertainties = [
        {'distribution': 'beta', 'alpha': 2, 'beta': 18},  # defect_rate1
        {'distribution': 'beta', 'alpha': 2, 'beta': 18},  # defect_rate2  
        {'distribution': 'normal', 'mean': 56, 'std': 2},   # market_price
        {'distribution': 'uniform', 'min': 1.8, 'max': 2.2}, # test_cost1
        {'distribution': 'uniform', 'min': 2.7, 'max': 3.3}  # test_cost2
    ]
    
    propagation_result = model.propagate_uncertainty(
        input_uncertainties, production_profit_function
    )
    
    # 测试敏感性分析
    sensitivity_result = model.sensitivity_analysis(
        base_values=[0.1, 0.1, 56, 2, 3],
        perturbation_size=0.1,
        evaluation_function=production_profit_function
    )
    
    # 生成报告
    analysis_results = {
        'bayesian_update': bayesian_result,
        'uncertainty_propagation': propagation_result,
        'sensitivity_analysis': sensitivity_result
    }
    
    model.create_error_propagation_report(analysis_results)
    model.visualize_uncertainty(propagation_result['samples'])
    
    print("误差传播模型测试完成！") 