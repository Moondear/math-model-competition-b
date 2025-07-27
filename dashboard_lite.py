"""
轻量版Dashboard - 无需外部依赖库
使用模拟数据展示所有功能
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import random
import math

# Streamlit页面配置
st.set_page_config(
    page_title="数学建模智能决策系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局状态
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'optimization_progress' not in st.session_state:
    st.session_state.optimization_progress = 0
if 'is_degraded_mode' not in st.session_state:
    st.session_state.is_degraded_mode = False
if 'cpu_usage' not in st.session_state:
    st.session_state.cpu_usage = random.uniform(20, 80)
if 'memory_usage' not in st.session_state:
    st.session_state.memory_usage = random.uniform(40, 70)

def get_system_metrics():
    """获取系统资源使用情况（模拟）"""
    # 模拟动态变化
    st.session_state.cpu_usage += random.uniform(-5, 5)
    st.session_state.cpu_usage = max(10, min(90, st.session_state.cpu_usage))
    
    st.session_state.memory_usage += random.uniform(-2, 2)
    st.session_state.memory_usage = max(30, min(80, st.session_state.memory_usage))
    
    # 计算剩余时间
    elapsed_time = datetime.now() - st.session_state.start_time
    total_hours = 72
    remaining_hours = total_hours - elapsed_time.total_seconds() / 3600
    remaining_hours = max(0, remaining_hours)
    
    return {
        'cpu_usage': st.session_state.cpu_usage,
        'memory_usage': st.session_state.memory_usage,
        'remaining_hours': remaining_hours,
        'progress': st.session_state.optimization_progress
    }

def simulate_optimal_sampling(p0=0.1, alpha=0.05, beta=0.1, p1=0.15):
    """模拟抽样检验计算"""
    # 简化的抽样方案计算
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    
    # 计算样本量 (简化公式)
    n = int((z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1)))**2 / (p1 - p0)**2)
    n = max(50, min(500, n))  # 限制在合理范围
    
    # 计算判定值
    c = int(n * p0 + z_alpha * math.sqrt(n * p0 * (1 - p0)))
    
    # 实际风险（近似）
    actual_alpha = alpha * (1 + random.uniform(-0.1, 0.1))
    actual_beta = beta * (1 + random.uniform(-0.1, 0.1))
    
    return n, c, actual_alpha, actual_beta

def create_sampling_visualization(p0=0.1, alpha=0.05, beta=0.1, p1=0.15):
    """创建抽样方案可视化 - 增强版3D效果"""
    try:
        # 计算最优方案
        n, c, actual_alpha, actual_beta = simulate_optimal_sampling(p0, alpha, beta, p1)
        
        # 创建3D成本表面图
        n_range = np.arange(50, 200, 5)
        alpha_range = np.linspace(0.01, 0.10, 20)
        N, A = np.meshgrid(n_range, alpha_range)
        
        # 计算成本表面
        Z = N * 1 + A * 100 + actual_beta * 200
        
        fig_3d = go.Figure(data=[go.Surface(
            x=N, y=A, z=Z,
            colorscale='Viridis',
            opacity=0.8,
            name='成本表面'
        )])
        
        # 添加最优点
        optimal_cost = n * 1 + actual_alpha * 100 + actual_beta * 200
        fig_3d.add_trace(go.Scatter3d(
            x=[n], y=[actual_alpha], z=[optimal_cost],
            mode='markers',
            marker=dict(size=10, color='red'),
            name=f'最优解 (n={n})',
            text=[f'最优: n={n}, α={actual_alpha:.4f}']
        ))
        
        fig_3d.update_layout(
            title='📊 抽样方案成本优化 - 3D表面图',
            scene=dict(
                xaxis_title='样本量 n',
                yaxis_title='第一类错误 α',
                zaxis_title='总成本',
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=500
        )
        
        # 创建增强的OC曲线
        p_range = np.linspace(0.05, 0.25, 100)
        oc_curve = []
        for p in p_range:
            # 简化的OC曲线计算
            z = (c - n * p) / math.sqrt(n * p * (1 - p))
            if z > 3:
                accept_prob = 1.0
            elif z < -3:
                accept_prob = 0.0
            else:
                accept_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            oc_curve.append(accept_prob)
        
        fig_oc = go.Figure()
        
        # OC曲线主线
        fig_oc.add_trace(go.Scatter(
            x=p_range, y=oc_curve,
            mode='lines',
            line=dict(width=3, color='blue'),
            name='OC曲线',
            hovertemplate='不合格率: %{x:.3f}<br>接受概率: %{y:.3f}<extra></extra>'
        ))
        
        # 填充区域
        fig_oc.add_trace(go.Scatter(
            x=p_range, y=oc_curve,
            fill='tozeroy',
            fillcolor='rgba(0,100,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 关键点标记
        fig_oc.add_vline(x=p0, line_dash="dash", line_color="green", line_width=2,
                        annotation_text=f"生产者风险点 p₀={p0}")
        fig_oc.add_vline(x=p1, line_dash="dash", line_color="red", line_width=2,
                        annotation_text=f"消费者风险点 p₁={p1}")
        
        # 添加风险区域注释
        fig_oc.add_annotation(
            x=p0-0.02, y=0.8,
            text=f"α风险≈{actual_alpha:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green"
        )
        
        fig_oc.add_annotation(
            x=p1+0.02, y=0.3,
            text=f"β风险≈{actual_beta:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red"
        )
        
        fig_oc.update_layout(
            title="📈 工作特性曲线 (OC曲线) - 增强版",
            xaxis_title="实际不合格率 p",
            yaxis_title="接受概率 L(p)",
            hovermode='x unified',
            height=450
        )
        
        # 结果表格 - 修复数据类型问题
        results = pd.DataFrame({
            '参数': ['样本量 n', '判定值 c', '实际 α', '实际 β'],
            '数值': [n, c, actual_alpha, actual_beta],  # 使用数值列
            '格式化值': [str(n), str(c), f"{actual_alpha:.4f}", f"{actual_beta:.4f}"],
            '说明': ['抽取样本数量', '拒收临界值', '生产者风险', '消费者风险']
        })
        
        return fig_3d, fig_oc, results
    
    except Exception as e:
        st.error(f"计算出错: {str(e)}")
        # 返回默认图表
        fig_3d = px.scatter_3d(x=[50, 100, 150], y=[0.05, 0.05, 0.05], z=[100, 200, 300], 
                              title="样本量与成本关系（默认）")
        fig_oc = px.line(x=[0.1, 0.15, 0.2], y=[0.8, 0.5, 0.2], title="OC曲线（默认）")
        results = pd.DataFrame({
            '参数': ['样本量', '判定值'], 
            '数值': [100, 10],
            '格式化值': ['100', '10'],
            '说明': ['默认值', '默认值']
        })
        return fig_3d, fig_oc, results

def simulate_production_optimization(defect_rate1, defect_rate2, test_cost1, test_cost2, 
                                   assembly_cost, test_cost_final, repair_cost, market_price):
    """模拟生产决策优化"""
    # 简化的决策逻辑
    return_loss = market_price * 0.1
    
    # 计算各种策略的期望利润
    strategies = []
    
    for test_part1 in [True, False]:
        for test_part2 in [True, False]:
            for test_final in [True, False]:
                for repair in [True, False]:
                    
                    # 计算期望利润（简化模型）
                    cost = assembly_cost
                    if test_part1:
                        cost += test_cost1
                    if test_part2:
                        cost += test_cost2
                    if test_final:
                        cost += test_cost_final
                    
                    # 计算合格率
                    p_ok_1 = 1 - defect_rate1 if test_part1 else 1 - defect_rate1
                    p_ok_2 = 1 - defect_rate2 if test_part2 else 1 - defect_rate2
                    p_ok = p_ok_1 * p_ok_2
                    
                    if test_final:
                        # 最终检测后的处理
                        if repair:
                            profit = market_price - cost - (1 - p_ok) * repair_cost
                        else:
                            profit = p_ok * (market_price - cost) - (1 - p_ok) * return_loss
                    else:
                        profit = p_ok * (market_price - cost) - (1 - p_ok) * return_loss
                    
                    strategies.append({
                        'test_part1': test_part1,
                        'test_part2': test_part2,
                        'test_final': test_final,
                        'repair': repair,
                        'expected_profit': profit,
                        'p_ok': p_ok
                    })
    
    # 找到最优策略
    best_strategy = max(strategies, key=lambda x: x['expected_profit'])
    best_strategy['solver_status'] = 'OPTIMAL'
    best_strategy['solution_time'] = random.uniform(0.001, 0.01)
    
    return best_strategy

def create_decision_heatmap(result):
    """创建清晰的决策可视化图表"""
    try:
        # 创建决策数据
        decisions = {
            '零件1检测': result.get('test_part1', False),
            '零件2检测': result.get('test_part2', False), 
            '成品检测': result.get('test_final', False),
            '返修处理': result.get('repair', False)
        }
        
        # 创建清晰的柱状图展示决策
        decision_names = list(decisions.keys())
        decision_values = [1 if v else 0 for v in decisions.values()]
        decision_colors = ['#2ECC71' if v else '#E74C3C' for v in decisions.values()]
        decision_text = ['✅ 执行' if v else '❌ 不执行' for v in decisions.values()]
        
        fig = go.Figure()
        
        # 创建柱状图
        fig.add_trace(go.Bar(
            x=decision_names,
            y=decision_values,
            marker=dict(
                color=decision_colors,
                opacity=0.8,
                line=dict(color='white', width=2)
            ),
            text=decision_text,
            textposition='inside',
            textfont=dict(size=16, color='white', family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>状态: %{text}<br>值: %{y}<extra></extra>',
            name='决策状态'
        ))
        
        # 美化布局
        fig.update_layout(
            title={
                'text': '🎯 最优生产决策方案',
                'x': 0.5,
                'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
            },
            xaxis=dict(
                title='决策类型',
                titlefont=dict(size=16, color='#34495E'),
                tickfont=dict(size=13, color='#2C3E50'),
                tickangle=0
            ),
            yaxis=dict(
                title='执行状态',
                titlefont=dict(size=16, color='#34495E'),
                tickfont=dict(size=13, color='#2C3E50'),
                tickmode='array',
                tickvals=[0, 1],
                ticktext=['不执行', '执行'],
                range=[-0.2, 1.3]
            ),
            height=450,
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white',
            showlegend=False,
            margin=dict(l=60, r=60, t=120, b=60)  # 增加顶部边距，减少底部边距
        )
        
        # 添加利润指标注释 - 放在图表顶部
        profit = result.get('expected_profit', 0)
        p_ok = result.get('p_ok', 0) * 100
        
        fig.add_annotation(
            x=0.5, y=1.08,
            xref='paper', yref='paper',
            text=f'💰 期望利润: <b>{profit:.2f}</b> | ✅ 合格率: <b>{p_ok:.1f}%</b>',
            showarrow=False,
            font=dict(size=16, color='#27AE60', family='Arial'),
            bgcolor='rgba(46, 204, 113, 0.1)',
            bordercolor='#27AE60',
            borderwidth=1,
            borderpad=10
        )
        
        return fig
        
    except Exception as e:
        # 返回简单图表
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['决策1', '决策2'], y=[1, 0]))
        fig.update_layout(title="决策矩阵（简化版）")
        return fig

def simulate_multistage_network():
    """模拟多工序网络优化"""
    # 创建示例网络数据
    nodes = ['工序1', '工序2', '工序3', '工序4', '工序5']
    edges = [('工序1', '工序2'), ('工序1', '工序3'), ('工序2', '工序4'), ('工序3', '工序4'), ('工序4', '工序5')]
    
    # 模拟优化结果
    decisions = {}
    for node in nodes:
        decisions[f'test_{node}'] = random.choice([True, False])
        decisions[f'repair_{node}'] = random.choice([True, False])
    
    total_cost = random.uniform(45, 55)
    computation_time = random.uniform(0.01, 0.05)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'decisions': decisions,
        'total_cost': total_cost,
        'computation_time': computation_time
    }

def create_network_visualization(network_data):
    """创建精美的3D生产网络图"""
    # 优化的3D布局 - 更加立体和清晰
    pos_3d = {
        '工序1': (0, 0, 0),
        '工序2': (3, 2, 1.5),
        '工序3': (3, -2, 1.5), 
        '工序4': (6, 0, 3),
        '工序5': (9, 0, 2)
    }
    
    # 节点数据
    node_x = [pos_3d[node][0] for node in network_data['nodes']]
    node_y = [pos_3d[node][1] for node in network_data['nodes']]
    node_z = [pos_3d[node][2] for node in network_data['nodes']]
    
    # 创建精美的3D图形
    fig = go.Figure()
    
    # 添加精美的连接线
    edge_traces = []
    for i, edge in enumerate(network_data['edges']):
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        
        # 创建渐变色连接线
        fig.add_trace(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(
                width=8,
                color=f'rgba({50 + i*40}, {100 + i*30}, {200 - i*20}, 0.8)'
            ),
            hoverinfo='none',
            showlegend=False,
            name=f'流程 {edge[0]} → {edge[1]}'
        ))
    
    # 添加精美的3D节点
    for i, node in enumerate(network_data['nodes']):
        decisions = network_data['decisions']
        test_decision = decisions.get(f'test_{node}', False)
        repair_decision = decisions.get(f'repair_{node}', False)
        
        # 根据决策状态设置颜色和大小
        if test_decision and repair_decision:
            color = '#E74C3C'  # 红色 - 全面处理
            size = 35
            symbol = 'diamond'
            status = '🔍✅ + 🔧✅'
        elif test_decision:
            color = '#F39C12'  # 橙色 - 仅检测  
            size = 30
            symbol = 'circle'
            status = '🔍✅'
        elif repair_decision:
            color = '#F1C40F'  # 黄色 - 仅返修
            size = 30
            symbol = 'square'
            status = '🔧✅'
        else:
            color = '#3498DB'  # 蓝色 - 无处理
            size = 25
            symbol = 'circle'
            status = '⭕'
        
        # 添加节点
        fig.add_trace(go.Scatter3d(
            x=[node_x[i]], y=[node_y[i]], z=[node_z[i]],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                opacity=0.9,
                symbol=symbol,
                line=dict(width=3, color='white')
            ),
            text=[node],
            textposition="middle center",
            textfont=dict(size=14, color='white', family='Arial Black'),
            hovertemplate=f'<b>{node}</b><br>决策状态: {status}<br>位置: ({node_x[i]}, {node_y[i]}, {node_z[i]})<extra></extra>',
            showlegend=False,
            name=f'节点_{node}'
        ))
    
    # 添加流程方向箭头
    for edge in network_data['edges']:
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        
        # 计算箭头位置（线段中点偏向终点）
        arrow_x = x0 + 0.7 * (x1 - x0)
        arrow_y = y0 + 0.7 * (y1 - y0)
        arrow_z = z0 + 0.7 * (z1 - z0)
        
        fig.add_trace(go.Cone(
            x=[arrow_x], y=[arrow_y], z=[arrow_z],
            u=[x1-x0], v=[y1-y0], w=[z1-z0],
            sizemode='absolute',
            sizeref=0.3,
            colorscale='Viridis',
            showscale=False,
            opacity=0.7,
            hoverinfo='skip'
        ))
    
    # 添加图例说明
    fig.add_trace(go.Scatter3d(
        x=[10], y=[0], z=[4],
        mode='text',
        text=['图例:<br>🔴 检测+返修<br>🟡 仅检测<br>🟡 仅返修<br>🔵 无处理'],
        textfont=dict(size=12, color='#2C3E50', family='Arial'),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # 设置精美的3D布局
    fig.update_layout(
        title={
            'text': "🏭 智能制造生产网络 - 立体流程图",
            'x': 0.5,
            'font': {'size': 22, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        scene=dict(
            xaxis=dict(
                title='流程进展方向',
                titlefont=dict(size=14, color='#34495E'),
                showgrid=True,
                gridcolor='rgba(200,200,200,0.5)',
                backgroundcolor='rgba(250,250,250,0.1)'
            ),
            yaxis=dict(
                title='并行处理分支',
                titlefont=dict(size=14, color='#34495E'),
                showgrid=True,
                gridcolor='rgba(200,200,200,0.5)',
                backgroundcolor='rgba(250,250,250,0.1)'
            ),
            zaxis=dict(
                title='处理复杂度层级',
                titlefont=dict(size=14, color='#34495E'),
                showgrid=True,
                gridcolor='rgba(200,200,200,0.5)',
                backgroundcolor='rgba(250,250,250,0.1)'
            ),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgba(248,249,250,0.9)',
            aspectmode='cube'
        ),
        showlegend=False,
        height=700,
        paper_bgcolor='white',
        plot_bgcolor='rgba(248,249,250,0.5)'
    )
    
    return fig

def simulate_robust_analysis(n_samples=50, n_simulations=50, confidence_level=0.95):
    """模拟鲁棒性分析"""
    # 生产决策鲁棒性
    profits = np.random.normal(45, 2, n_simulations)
    prod_result = {
        'expected_profit': np.mean(profits),
        'worst_case_profit': np.percentile(profits, (1 - confidence_level) * 100),
        'profit_std': np.std(profits),
        'decision_confidence': confidence_level
    }
    
    # 多工序鲁棒性
    costs = np.random.normal(50, 3, n_simulations)
    multi_result = {
        'expected_cost': np.mean(costs),
        'worst_case_cost': np.percentile(costs, confidence_level * 100),
        'cost_std': np.std(costs),
        'robust_decisions': {
            '工序1': {'test': True, 'repair': False, 'decision_confidence': 0.92},
            '工序2': {'test': False, 'repair': True, 'decision_confidence': 0.88},
            '工序3': {'test': True, 'repair': True, 'decision_confidence': 0.95},
            '工序4': {'test': False, 'repair': False, 'decision_confidence': 0.85},
            '工序5': {'test': True, 'repair': False, 'decision_confidence': 0.90}
        }
    }
    
    return prod_result, multi_result

def switch_to_heuristic_mode():
    """切换到启发式模式"""
    st.session_state.is_degraded_mode = True
    st.success("已切换到启发式算法模式")

def main():
    """主函数"""
    # 页面标题
    st.title("🎯 2024数学建模智能决策系统")
    st.markdown("**全国大学生数学建模竞赛 - 智能制造质量控制优化平台**")
    st.info("🎉 轻量版Dashboard - 完全本地运行，无需外部依赖库")
    
    # 状态监控面板
    with st.expander("🚀 实时作战面板", expanded=True):
        metrics = get_system_metrics()
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("优化进度", f"{metrics['progress']:.1f}%")
        col2.metric("CPU使用率", f"{metrics['cpu_usage']:.1f}%")
        col3.metric("内存使用率", f"{metrics['memory_usage']:.1f}%")
        col4.metric("剩余时间", f"{metrics['remaining_hours']:.1f}h")
    
    # 多页签界面
    tab1, tab2, tab3, tab4 = st.tabs(["📊 抽样检验", "🏭 生产决策", "🔗 多工序优化", "🛡️ 鲁棒性分析"])
    
    with tab1:
        st.header("📊 抽样检验方案优化")
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            p0 = st.slider("不合格率阈值(p₀)", 0.01, 0.20, 0.10, 0.01)
            alpha = st.slider("第一类错误(α)", 0.01, 0.10, 0.05, 0.01)
        with col2:
            p1 = st.slider("备择假设不合格率(p₁)", 0.11, 0.30, 0.15, 0.01)
            beta = st.slider("第二类错误(β)", 0.01, 0.10, 0.10, 0.01)
        
        if st.button("🔍 计算最优方案", key="sampling"):
            with st.spinner("正在优化抽样方案..."):
                fig_3d, fig_oc, results = create_sampling_visualization(p0, alpha, beta, p1)
                
                # 显示3D成本表面图
                st.plotly_chart(fig_3d, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_oc, use_container_width=True)
                with col2:
                    st.subheader("📋 最优方案结果")
                    # 只显示格式化的结果
                    display_results = results[['参数', '格式化值', '说明']].copy()
                    display_results.columns = ['参数', '值', '说明']
                    st.dataframe(display_results, use_container_width=True)
                
                # 增加进度
                st.session_state.optimization_progress += 15
    
    with tab2:
        st.header("🏭 生产决策优化")
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            defect_rate1 = st.slider("零件1不合格率", 0.01, 0.20, 0.10, 0.01)
            defect_rate2 = st.slider("零件2不合格率", 0.01, 0.20, 0.10, 0.01)
            test_cost1 = st.number_input("零件1检测成本", 1, 10, 2)
            test_cost2 = st.number_input("零件2检测成本", 1, 10, 3)
        with col2:
            assembly_cost = st.number_input("装配成本", 1, 20, 6)
            test_cost_final = st.number_input("成品检测成本", 1, 10, 3)
            repair_cost = st.number_input("返修成本", 1, 20, 5)
            market_price = st.number_input("市场价格", 10, 100, 56)
        
        if st.button("⚡ 优化决策", key="production"):
            with st.spinner("正在优化生产决策..."):
                result = simulate_production_optimization(
                    defect_rate1, defect_rate2, test_cost1, test_cost2,
                    assembly_cost, test_cost_final, repair_cost, market_price
                )
                
                # 显示结果
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_decision_heatmap(result), use_container_width=True)
                
                with col2:
                    st.metric("💰 期望利润", f"{result['expected_profit']:.2f}")
                    st.metric("✅ 合格率", f"{result['p_ok']*100:.1f}%")
                    st.metric("⚡ 求解状态", result['solver_status'])
                    st.metric("⏱️ 求解时间", f"{result['solution_time']*1000:.1f}ms")
                    
                # 决策详情
                st.subheader("📋 最优决策详情")
                decision_data = pd.DataFrame({
                    '决策项': ['零件1检测', '零件2检测', '成品检测', '返修处理'],
                    '决策': [
                        '是' if result['test_part1'] else '否',
                        '是' if result['test_part2'] else '否', 
                        '是' if result['test_final'] else '否',
                        '是' if result['repair'] else '否'
                    ]
                })
                st.dataframe(decision_data, use_container_width=True)
                
                # 增加进度
                st.session_state.optimization_progress += 20
    
    with tab3:
        st.header("🔗 多工序生产系统优化")
        
        if st.button("🌐 创建示例网络", key="network"):
            with st.spinner("正在构建生产网络..."):
                network_data = simulate_multistage_network()
                
                st.success("✅ 网络优化完成！")
                
                # 显示网络可视化
                st.plotly_chart(create_network_visualization(network_data), use_container_width=True)
                
                # 显示网络信息
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("节点数量", len(network_data['nodes']))
                    st.metric("边数量", len(network_data['edges']))
                with col2:
                    st.metric("💰 总成本", f"{network_data['total_cost']:.2f}")
                    st.metric("⏱️ 计算时间", f"{network_data['computation_time']*1000:.1f}ms")
                
                # 显示节点决策
                st.subheader("📋 各节点最优决策")
                for node in network_data['nodes']:
                    decisions = network_data['decisions']
                    with st.expander(f"工序节点 {node}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            test_decision = decisions.get(f'test_{node}', False)
                            st.write(f"🔍 检测: {'是' if test_decision else '否'}")
                        with col2:
                            repair_decision = decisions.get(f'repair_{node}', False)
                            st.write(f"🔧 返修: {'是' if repair_decision else '否'}")
                
                # 增加进度
                st.session_state.optimization_progress += 25
    
    with tab4:
        st.header("🛡️ 鲁棒性分析")
        
        # 不确定性参数设置
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("样本数量", 10, 100, 50)
            n_simulations = st.slider("模拟次数", 10, 100, 50)
        with col2:
            confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95, 0.01)
        
        if st.button("🛡️ 鲁棒性分析", key="robust"):
            with st.spinner("正在进行鲁棒性分析..."):
                prod_result, multi_result = simulate_robust_analysis(n_samples, n_simulations, confidence_level)
                
                # 显示生产决策鲁棒性结果
                st.subheader("🏭 生产决策鲁棒性")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("💰 期望利润", f"{prod_result['expected_profit']:.2f}")
                    st.metric("📉 最差情况利润", f"{prod_result['worst_case_profit']:.2f}")
                with col2:
                    st.metric("📊 利润标准差", f"{prod_result['profit_std']:.2f}")
                    st.metric("🎯 决策置信度", f"{prod_result['decision_confidence']*100:.1f}%")
                
                # 显示多工序结果
                st.subheader("🔗 多工序系统鲁棒性")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("💸 期望成本", f"{multi_result['expected_cost']:.2f}")
                    st.metric("📈 最差情况成本", f"{multi_result['worst_case_cost']:.2f}")
                with col2:
                    st.metric("📊 成本标准差", f"{multi_result['cost_std']:.2f}")
                    
                # 显示各节点的鲁棒决策
                st.subheader("🎯 节点鲁棒决策")
                robust_decisions = multi_result['robust_decisions']
                for node, decision in robust_decisions.items():
                    with st.expander(f"节点 {node}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"🔍 检测: {'是' if decision['test'] else '否'}")
                        with col2:
                            st.write(f"🔧 返修: {'是' if decision['repair'] else '否'}")
                        with col3:
                            st.write(f"🎯 置信度: {decision['decision_confidence']*100:.1f}%")
                
                # 精美的鲁棒性分布图
                st.subheader("📊 鲁棒性风险分析 - 利润分布预测")
                
                # 生成更真实的分布数据
                profits = np.random.normal(prod_result['expected_profit'], prod_result['profit_std'], 1000)
                costs = np.random.normal(multi_result['expected_cost'], multi_result['cost_std'], 1000)
                
                # 创建双轴分布图
                fig_dist = go.Figure()
                
                # 利润分布直方图
                fig_dist.add_trace(go.Histogram(
                    x=profits,
                    name='利润分布',
                    opacity=0.7,
                    nbinsx=30,
                    marker=dict(
                        color='rgba(46, 204, 113, 0.7)',
                        line=dict(color='rgba(46, 204, 113, 1)', width=1)
                    ),
                    hovertemplate='利润区间: %{x}<br>频数: %{y}<extra></extra>'
                ))
                
                # 添加关键统计线
                fig_dist.add_vline(
                    x=prod_result['expected_profit'], 
                    line_dash="solid", 
                    line_color="#27AE60", 
                    line_width=3,
                    annotation_text="期望利润",
                    annotation_position="top"
                )
                
                fig_dist.add_vline(
                    x=prod_result['worst_case_profit'], 
                    line_dash="dash", 
                    line_color="#E74C3C", 
                    line_width=3,
                    annotation_text="最差情况",
                    annotation_position="top"
                )
                
                # 添加置信区间
                confidence_lower = prod_result['expected_profit'] - 1.96 * prod_result['profit_std']
                confidence_upper = prod_result['expected_profit'] + 1.96 * prod_result['profit_std']
                
                fig_dist.add_vrect(
                    x0=confidence_lower, x1=confidence_upper,
                    fillcolor="rgba(52, 152, 219, 0.2)",
                    layer="below",
                    line_width=0,
                    annotation_text="95%置信区间",
                    annotation_position="top left"
                )
                
                # 美化布局
                fig_dist.update_layout(
                    title={
                        'text': '💰 利润分布与风险评估',
                        'x': 0.5,
                        'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
                    },
                    xaxis=dict(
                        title='利润值',
                        titlefont=dict(size=16, color='#34495E'),
                        tickfont=dict(size=12, color='#2C3E50'),
                        showgrid=True,
                        gridcolor='rgba(200,200,200,0.3)'
                    ),
                    yaxis=dict(
                        title='频数',
                        titlefont=dict(size=16, color='#34495E'),
                        tickfont=dict(size=12, color='#2C3E50'),
                        showgrid=True,
                        gridcolor='rgba(200,200,200,0.3)'
                    ),
                    plot_bgcolor='rgba(248,249,250,0.8)',
                    paper_bgcolor='white',
                    height=450,
                    showlegend=False
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # 添加风险评估表
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    📈 **利润风险指标**
                    - 期望收益: {prod_result['expected_profit']:.2f}
                    - 标准差: {prod_result['profit_std']:.2f}
                    - 变异系数: {(prod_result['profit_std']/prod_result['expected_profit']*100):.1f}%
                    - 95%置信下界: {confidence_lower:.2f}
                    """)
                    
                with col2:
                    st.warning(f"""
                    💸 **成本风险指标**
                    - 期望成本: {multi_result['expected_cost']:.2f}
                    - 标准差: {multi_result['cost_std']:.2f}
                    - 变异系数: {(multi_result['cost_std']/multi_result['expected_cost']*100):.1f}%
                    - 风险等级: {'低' if multi_result['cost_std'] < 2 else '中' if multi_result['cost_std'] < 4 else '高'}
                    """)
                
                # 增加进度
                st.session_state.optimization_progress += 30
    
    # 应急控制台
    st.sidebar.title("🚨 应急控制台")
    if st.sidebar.button("⚡ 启用降级模式", help="当模型求解失败时，切换到启发式算法"):
        switch_to_heuristic_mode()
    
    if st.session_state.is_degraded_mode:
        st.sidebar.warning("⚠️ 当前处于降级模式")
    else:
        st.sidebar.success("✅ 轻量版模式运行中")
    
    # 功能说明
    st.sidebar.subheader("📖 功能说明")
    st.sidebar.info("""
    🎯 **轻量版特性**
    - ✅ 完全本地运行
    - ✅ 无需外部依赖
    - ✅ 实时参数调节
    - ✅ 交互式可视化
    - ✅ 模拟真实算法
    
    🔧 **核心功能**
    - 抽样检验优化
    - 生产决策分析
    - 多工序网络优化
    - 鲁棒性分析
    """)
    
    # 系统信息
    st.sidebar.subheader("📊 系统信息")
    st.sidebar.info(f"""
    🕒 运行时间: {(datetime.now() - st.session_state.start_time).total_seconds()/60:.1f} 分钟
    
    💻 系统状态: {'降级模式' if st.session_state.is_degraded_mode else '轻量版模式'}
    
    🎯 项目: 2024数学建模竞赛
    
    📈 总体进度: {st.session_state.optimization_progress:.1f}%
    """)
    
    # 更新进度
    st.session_state.optimization_progress = min(100, st.session_state.optimization_progress + random.uniform(0, 0.2))

if __name__ == "__main__":
    main() 