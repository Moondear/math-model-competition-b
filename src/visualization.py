"""
可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, beta
import networkx as nx
import os
from .font_utils import setup_chinese_font, ensure_output_dir

def plot_sampling_distribution(n: int, c: int, p0: float, p1: float) -> str:
    """绘制抽样分布图
    
    Args:
        n: 样本量
        c: 判定值
        p0: 原假设不良率
        p1: 备择假设不良率
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建x轴数据点
    x = np.arange(0, n+1)
    
    # 计算两个分布
    y0 = binom.pmf(x, n, p0)
    y1 = binom.pmf(x, n, p1)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制分布曲线
    plt.plot(x, y0, 'b-', label=f'H0: p={p0}', alpha=0.7)
    plt.plot(x, y1, 'r-', label=f'H1: p={p1}', alpha=0.7)
    
    # 添加判定线
    plt.axvline(x=c, color='g', linestyle='--', label=f'判定值 c={c}')
    
    # 填充错误区域
    x_fill0 = np.arange(c+1, n+1)
    y_fill0 = binom.pmf(x_fill0, n, p0)
    plt.fill_between(x_fill0, y_fill0, alpha=0.3, color='blue', label='第一类错误 (α)')
    
    x_fill1 = np.arange(0, c+1)
    y_fill1 = binom.pmf(x_fill1, n, p1)
    plt.fill_between(x_fill1, y_fill1, alpha=0.3, color='red', label='第二类错误 (β)')
    
    # 设置标题和标签
    plt.title('抽样检验方案的概率分布')
    plt.xlabel('不合格品数量')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/sampling_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_error_rates(p0: float, p1: float, n: int, c: int) -> str:
    """绘制错误率曲线
    
    Args:
        p0: 原假设不良率
        p1: 备择假设不良率
        n: 样本量
        c: 判定值
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建不良率范围
    p_range = np.linspace(0, 0.3, 1000)
    
    # 计算工作特性曲线
    oc_curve = 1 - binom.cdf(c, n, p_range)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制工作特性曲线
    plt.plot(p_range, oc_curve, 'b-', label='工作特性曲线')
    
    # 添加关键点
    plt.plot(p0, 1-0.05, 'ro', label=f'H0: p={p0}')
    plt.plot(p1, 0.1, 'go', label=f'H1: p={p1}')
    
    # 添加辅助线
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.3)
    plt.axvline(x=p0, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=p1, color='g', linestyle='--', alpha=0.3)
    
    # 设置标题和标签
    plt.title('抽样检验方案的工作特性曲线')
    plt.xlabel('实际不良率')
    plt.ylabel('接受概率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/error_rates.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_sampling_dashboard(n: int, c: int, p0: float = 0.1, p1: float = 0.15) -> tuple[str, str]:
    """创建抽样方案的可视化面板
    
    Args:
        n: 样本量
        c: 判定值
        p0: 原假设不良率
        p1: 备择假设不良率
        
    Returns:
        tuple[str, str]: 保存的两个图表文件路径
    """
    # 绘制概率分布图
    dist_path = plot_sampling_distribution(n, c, p0, p1)
    
    # 绘制错误率曲线
    error_path = plot_error_rates(p0, p1, n, c)
    
    return dist_path, error_path 

def plot_production_decision(result: dict) -> str:
    """绘制生产决策流程图
    
    Args:
        result: 优化结果字典
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点
    nodes = {
        'part1': '零件1',
        'test1': '检测1',
        'part2': '零件2',
        'test2': '检测2',
        'assembly': '装配',
        'final_test': '成品检测',
        'repair': '拆解返修',
        'market': '市场'
    }
    
    # 设置节点位置
    pos = {
        'part1': (0, 3),
        'test1': (1, 3),
        'part2': (0, 1),
        'test2': (1, 1),
        'assembly': (2, 2),
        'final_test': (3, 2),
        'repair': (3, 1),
        'market': (4, 2)
    }
    
    # 添加节点和边
    for node, label in nodes.items():
        G.add_node(node, label=label)
    
    # 添加基本流程边
    edges = [
        ('part1', 'test1'),
        ('part2', 'test2'),
        ('test1', 'assembly'),
        ('test2', 'assembly'),
        ('assembly', 'final_test'),
        ('final_test', 'market'),
        ('final_test', 'repair'),
        ('repair', 'assembly')
    ]
    
    # 设置边的颜色
    edge_colors = []
    for u, v in edges:
        if (u == 'test1' and v == 'assembly' and result['test_part1']) or \
           (u == 'test2' and v == 'assembly' and result['test_part2']) or \
           (u == 'assembly' and v == 'final_test' and result['test_final']) or \
           (u == 'final_test' and v == 'repair' and result['repair']):
            edge_colors.append('red')
        else:
            edge_colors.append('gray')
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                          arrows=True, width=2, arrowsize=20)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=2000, alpha=0.6)
    
    # 添加标签
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'),
                           font_size=10)
    
    # 添加标题和说明
    plt.title('生产决策流程图\n' + 
             f'期望利润: {result["expected_profit"]:.2f}, ' +
             f'合格率: {result["p_ok"]:.2%}')
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/production_decision.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_cost_breakdown(result: dict, params: dict) -> str:
    """绘制成本收益分析图
    
    Args:
        result: 优化结果字典
        params: 生产参数字典
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 计算各项成本和收益
    test_cost1 = params.test_cost1 if result['test_part1'] else 0
    test_cost2 = params.test_cost2 if result['test_part2'] else 0
    test_cost_final = params.test_cost_final if result['test_final'] else 0
    repair_cost = params.repair_cost * (1 - result['p_ok']) if result['repair'] else 0
    return_loss = params.return_loss * (1 - result['p_ok']) if not result['repair'] else 0
    revenue = params.market_price * result['p_ok']
    
    # 创建数据
    categories = ['检测成本', '装配成本', '返修成本', '退货损失', '收入', '净利润']
    values = [
        test_cost1 + test_cost2 + test_cost_final,
        params.assembly_cost,
        repair_cost,
        return_loss,
        revenue,
        result['expected_profit']
    ]
    colors = ['#FF9999', '#99FF99', '#99CCFF', '#FFCC99', '#99FFCC', '#FF99CC']
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制条形图
    bars = plt.bar(categories, values, color=colors)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # 设置标题和标签
    plt.title('生产决策成本收益分析')
    plt.xlabel('类别')
    plt.ylabel('金额')
    
    # 调整布局
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/cost_breakdown.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_production_dashboard(result: dict, params: dict) -> tuple[str, str]:
    """创建生产决策的可视化面板
    
    Args:
        result: 优化结果字典
        params: 生产参数字典
        
    Returns:
        tuple[str, str]: 保存的两个图表文件路径
    """
    # 绘制决策流程图
    decision_path = plot_production_decision(result)
    
    # 绘制成本收益分析图
    cost_path = plot_cost_breakdown(result, params)
    
    return decision_path, cost_path 

def plot_production_network(graph: nx.DiGraph, result: dict) -> str:
    """绘制生产网络图
    
    Args:
        graph: 生产网络
        result: 优化结果
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 设置节点位置
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_color='gray',
                          arrows=True, width=1, arrowsize=20)
    
    # 准备节点颜色和大小
    node_colors = []
    node_sizes = []
    for node in graph.nodes:
        decision = result['decisions'][node]
        # 根据检测和返修决策设置颜色
        if decision['test'] and decision['repair']:
            color = 'lightcoral'  # 检测且返修
        elif decision['test']:
            color = 'lightblue'   # 仅检测
        else:
            color = 'lightgreen'  # 不检测
        node_colors.append(color)
        # 根据成本设置大小
        size = 2000 + decision['cost'] * 100
        node_sizes.append(size)
    
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                          node_size=node_sizes, alpha=0.6)
    
    # 添加节点标签
    labels = {}
    for node in graph.nodes:
        decision = result['decisions'][node]
        labels[node] = f"{node}\n合格率: {decision['p_ok']:.2%}\n成本: {decision['cost']:.1f}"
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='不检测',
                  markerfacecolor='lightgreen', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='仅检测',
                  markerfacecolor='lightblue', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='检测且返修',
                  markerfacecolor='lightcoral', markersize=15)
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # 添加标题
    plt.title('生产网络优化结果\n' +
             f'总成本: {result["total_cost"]:.2f}, ' +
             f'求解状态: {result["solver_status"]}')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/production_network.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_cost_distribution(result: dict, graph: nx.DiGraph) -> str:
    """绘制成本分布图
    
    Args:
        result: 优化结果
        graph: 生产网络
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 收集数据
    nodes = list(result['decisions'].keys())
    costs = [result['decisions'][node]['cost'] for node in nodes]
    
    # 创建堆叠条形图的数据
    process_costs = []
    test_costs = []
    repair_costs = []
    
    for node in nodes:
        decision = result['decisions'][node]
        params = graph.nodes[node]['params']
        
        # 基础加工成本
        process_costs.append(params.process_cost)
        
        # 检测成本
        test_cost = params.test_cost if decision['test'] else 0
        test_costs.append(test_cost)
        
        # 返修成本
        repair_cost = (params.repair_cost * (1 - decision['p_ok']) 
                      if decision['repair'] else 0)
        repair_costs.append(repair_cost)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制堆叠条形图
    width = 0.35
    plt.bar(nodes, process_costs, width, label='加工成本')
    plt.bar(nodes, test_costs, width, bottom=process_costs, label='检测成本')
    plt.bar(nodes, repair_costs, width, 
            bottom=np.array(process_costs) + np.array(test_costs),
            label='返修成本')
    
    # 设置标题和标签
    plt.title('各节点成本构成')
    plt.xlabel('节点')
    plt.ylabel('成本')
    
    # 添加图例
    plt.legend()
    
    # 旋转x轴标签
    plt.xticks(rotation=45)
    
    # 添加网格线
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/cost_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_multistage_dashboard(graph: nx.DiGraph, result: dict) -> tuple[str, str]:
    """创建多工序优化的可视化面板
    
    Args:
        graph: 生产网络
        result: 优化结果
        
    Returns:
        tuple[str, str]: 保存的两个图表文件路径
    """
    # 绘制生产网络图
    network_path = plot_production_network(graph, result)
    
    # 绘制成本分布图
    cost_path = plot_cost_distribution(result, graph)
    
    return network_path, cost_path 

def plot_uncertainty_distribution(p_hat: float, n: int, 
                                confidence_level: float = 0.95) -> str:
    """绘制不确定性分布图
    
    Args:
        p_hat: 观测的不合格率
        n: 样本量
        confidence_level: 置信水平
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 生成Beta分布参数
    k = int(p_hat * n)
    alpha = k + 1
    beta_param = n - k + 1
    
    # 生成x轴数据点
    x = np.linspace(0, min(1, p_hat * 3), 1000)
    
    # 计算Beta分布概率密度
    y = beta.pdf(x, alpha, beta_param)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制概率密度曲线
    plt.plot(x, y, 'b-', label='Beta分布')
    
    # 添加观测值和置信区间
    plt.axvline(x=p_hat, color='r', linestyle='--', 
                label=f'观测值 p={p_hat:.3f}')
    
    # 计算置信区间
    ci_upper = beta.ppf(confidence_level, alpha, beta_param)
    plt.axvline(x=ci_upper, color='g', linestyle='--',
                label=f'{confidence_level*100:.0f}%置信上界')
    
    # 填充置信区间
    x_fill = x[x <= ci_upper]
    y_fill = beta.pdf(x_fill, alpha, beta_param)
    plt.fill_between(x_fill, y_fill, alpha=0.3, color='gray',
                    label='置信区间')
    
    # 设置标题和标签
    plt.title(f'不合格率的Beta分布\n(n={n}, k={k})')
    plt.xlabel('不合格率')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/uncertainty_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_profit_distribution(profits: np.ndarray) -> str:
    """绘制利润分布图
    
    Args:
        profits: 利润数组
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 计算基本统计量
    mean_profit = np.mean(profits)
    min_profit = np.min(profits)
    max_profit = np.max(profits)
    std_profit = np.std(profits)
    
    # 设置bins数量，确保能显示出分布形状
    n_bins = min(30, len(np.unique(profits)))
    
    # 绘制直方图
    plt.hist(profits, bins=n_bins, density=True, alpha=0.7,
             color='blue', label='模拟结果')
    
    # 添加核密度估计（只在有足够不同的值时添加）
    if len(np.unique(profits)) > 5:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(profits)
        x = np.linspace(min_profit - std_profit, max_profit + std_profit, 200)
        plt.plot(x, kde(x), 'r-', label='核密度估计')
    
    # 添加统计指标的垂直线
    plt.axvline(x=mean_profit, color='g', linestyle='--',
                label=f'期望值: {mean_profit:.2f}')
    plt.axvline(x=min_profit, color='r', linestyle='--',
                label=f'最小值: {min_profit:.2f}')
    
    # 添加置信区间
    if len(np.unique(profits)) > 1:
        percentile_5 = np.percentile(profits, 5)
        percentile_95 = np.percentile(profits, 95)
        plt.axvspan(percentile_5, percentile_95, alpha=0.2, color='gray',
                   label='90%置信区间')
    
    # 设置坐标轴范围
    plt.xlim(min_profit - std_profit, max_profit + std_profit)
    
    # 如果所有值都相同，调整y轴范围使图形更合理
    if min_profit == max_profit:
        plt.ylim(0, 2)
        plt.text(mean_profit, 1, '所有模拟结果相同\n说明决策非常稳健',
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 设置标题和标签
    plt.title('利润分布\n' +
             f'标准差: {std_profit:.2f}, ' +
             f'变异系数: {(std_profit/mean_profit*100):.2f}%')
    plt.xlabel('利润')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/profit_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def plot_robust_network(graph: nx.DiGraph, result: dict) -> str:
    """绘制鲁棒生产网络图
    
    Args:
        graph: 生产网络
        result: 优化结果
        
    Returns:
        str: 保存的文件路径
    """
    # 设置中文字体
    setup_chinese_font()
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 设置节点位置
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_color='gray',
                          arrows=True, width=1, arrowsize=20)
    
    # 准备节点颜色和大小
    node_colors = []
    node_sizes = []
    for node in graph.nodes:
        decision = result['robust_decisions'][node]
        # 根据检测和返修决策设置颜色
        if decision['test'] and decision['repair']:
            color = 'lightcoral'  # 检测且返修
        elif decision['test']:
            color = 'lightblue'   # 仅检测
        else:
            color = 'lightgreen'  # 不检测
        node_colors.append(color)
        # 根据决策置信度设置大小
        size = 2000 + decision['decision_confidence'] * 2000
        node_sizes.append(size)
    
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                          node_size=node_sizes, alpha=0.6)
    
    # 添加节点标签
    labels = {}
    for node in graph.nodes:
        decision = result['robust_decisions'][node]
        labels[node] = f"{node}\n置信度: {decision['decision_confidence']:.1%}"
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='不检测',
                  markerfacecolor='lightgreen', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='仅检测',
                  markerfacecolor='lightblue', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='检测且返修',
                  markerfacecolor='lightcoral', markersize=15)
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # 添加标题
    plt.title('鲁棒生产网络\n' +
             f'期望成本: {result["expected_cost"]:.2f}, ' +
             f'最差情况: {result["worst_case_cost"]:.2f}')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    ensure_output_dir()
    filepath = 'output/robust_network.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def create_robust_dashboard(result: dict, base_params: dict = None,
                          graph: nx.DiGraph = None) -> tuple[str, ...]:
    """创建鲁棒优化的可视化面板
    
    Args:
        result: 优化结果
        base_params: 基准参数（用于生产决策优化）
        graph: 生产网络（用于多工序优化）
        
    Returns:
        tuple[str, ...]: 保存的图表文件路径
    """
    paths = []
    
    # 绘制不确定性分布
    if base_params:
        # 生产决策优化的不确定性
        paths.append(plot_uncertainty_distribution(
            base_params.defect_rate1, 100))
        # 使用simulation_results中的profit字段
        profits = [r['profit'] for r in result.get('simulation_results', [])]
        if profits:
            paths.append(plot_profit_distribution(np.array(profits)))
    
    # 绘制鲁棒网络
    if graph:
        # 多工序优化的网络图
        paths.append(plot_robust_network(graph, result))
    
    return tuple(paths) 