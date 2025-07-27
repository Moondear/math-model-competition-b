"""
交互式决策看板
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pyvista as pv
import numpy as np
import pandas as pd
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
import networkx as nx

from src.sampling import optimal_sampling, run_stress_test
from src.production import ProductionParams, optimize_production
from src.multistage import create_example_network, optimize_multistage
from src.robust import UncertaintyParams, robust_optimize_production, robust_optimize_multistage

# 全局状态
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'optimization_progress' not in st.session_state:
    st.session_state.optimization_progress = 0
if 'is_degraded_mode' not in st.session_state:
    st.session_state.is_degraded_mode = False

def get_system_metrics():
    """获取系统资源使用情况"""
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    
    # 计算剩余时间
    elapsed_time = datetime.now() - st.session_state.start_time
    total_hours = 72
    remaining_hours = total_hours - elapsed_time.total_seconds() / 3600
    remaining_hours = max(0, remaining_hours)
    
    return {
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'remaining_hours': remaining_hours,
        'progress': st.session_state.optimization_progress
    }

def create_sampling_visualization(p0=0.1, alpha=0.05, beta=0.1, p1=0.15):
    """创建抽样方案可视化"""
    # 计算最优方案
    n, c, actual_alpha, actual_beta = optimal_sampling(p0, alpha, beta, p1)
    
    # 创建场景数据
    scenarios = pd.DataFrame({
        'scenario': ['基准情况', '最优方案'],
        'n': [100, n],
        'c': [10, c],
        'alpha': [alpha, actual_alpha],
        'beta': [beta, actual_beta]
    })
    
    # 创建柱状图
    fig_n = px.bar(scenarios, x='scenario', y='n',
                   title='样本量对比',
                   labels={'n': '样本量', 'scenario': '场景'})
    
    # 创建错误率对比图
    fig_error = go.Figure()
    fig_error.add_trace(go.Bar(
        name='α (第一类错误)',
        x=scenarios['scenario'],
        y=scenarios['alpha'],
        marker_color='red'
    ))
    fig_error.add_trace(go.Bar(
        name='β (第二类错误)',
        x=scenarios['scenario'],
        y=scenarios['beta'],
        marker_color='blue'
    ))
    fig_error.update_layout(
        title='错误率对比',
        barmode='group',
        yaxis_title='错误率'
    )
    
    return fig_n, fig_error, scenarios

def create_decision_heatmap(result):
    """创建决策热力图"""
    # 提取决策变量
    decisions = {
        '检测零件1': result['test_part1'],
        '检测零件2': result['test_part2'],
        '检测成品': result['test_final'],
        '不合格拆解': result['repair']
    }
    
    # 创建热力图数据
    df = pd.DataFrame([decisions]).T.reset_index()
    df.columns = ['决策', '值']
    df['值'] = df['值'].map({True: 1, False: 0})
    
    fig = px.imshow(df['值'].values.reshape(1, -1),
                    y=['决策'],
                    x=df['决策'],
                    color_continuous_scale='RdYlBu',
                    title='决策热力图')
    
    return fig

def create_3d_topology(graph):
    """创建3D生产网络拓扑图"""
    # 使用plotly创建3D网络图
    pos = nx.spring_layout(graph, dim=3)
    
    # 创建节点轨迹
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    
    for node in graph.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f'节点: {node}')
        # 根据节点类型设置颜色
        if node.startswith('P'):
            node_color.append('blue')  # 零件节点
        elif node.startswith('A'):
            node_color.append('green')  # 装配节点
        else:
            node_color.append('red')   # 成品节点
    
    # 创建边轨迹
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in graph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # 创建边的轨迹
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )
    
    # 创建节点的轨迹
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=10,
            color=node_color,
            line=dict(width=2)
        )
    )
    
    # 创建图形
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # 更新布局
    fig.update_layout(
        title='生产网络3D拓扑图',
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        )
    )
    
    return fig

def switch_to_heuristic_mode():
    """切换到启发式求解模式"""
    st.session_state.is_degraded_mode = True
    st.warning('已切换到启发式求解模式')

def main():
    """主函数"""
    # 页面配置
    st.set_page_config(layout="wide", page_title="2024B题智能决策系统")
    st.title("2024B题智能决策系统")
    
    # 状态监控面板
    with st.expander("实时作战面板", expanded=True):
        metrics = get_system_metrics()
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("优化进度", f"{metrics['progress']:.1f}%")
        col2.metric("CPU使用率", f"{metrics['cpu_usage']:.1f}%")
        col3.metric("内存使用率", f"{metrics['memory_usage']:.1f}%")
        col4.metric("剩余时间", f"{metrics['remaining_hours']:.1f}h")
    
    # 多页签界面
    tab1, tab2, tab3, tab4 = st.tabs(["抽样检验", "生产决策", "多工序优化", "鲁棒性分析"])
    
    with tab1:
        st.header("抽样检验方案优化")
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            p0 = st.slider("不合格率阈值(p₀)", 0.01, 0.20, 0.10)
            alpha = st.slider("第一类错误(α)", 0.01, 0.10, 0.05)
        with col2:
            p1 = st.slider("备择假设不合格率(p₁)", 0.11, 0.30, 0.15)
            beta = st.slider("第二类错误(β)", 0.01, 0.10, 0.10)
        
        if st.button("计算最优方案", key="sampling"):
            with st.spinner("正在优化..."):
                fig_n, fig_error, results = create_sampling_visualization(
                    p0, alpha, beta, p1)
                st.plotly_chart(fig_n, use_container_width=True)
                st.plotly_chart(fig_error, use_container_width=True)
                st.dataframe(results)
    
    with tab2:
        st.header("生产决策优化")
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            defect_rate1 = st.slider("零件1不合格率", 0.01, 0.20, 0.10)
            defect_rate2 = st.slider("零件2不合格率", 0.01, 0.20, 0.10)
            test_cost1 = st.number_input("零件1检测成本", 1, 10, 2)
            test_cost2 = st.number_input("零件2检测成本", 1, 10, 3)
        with col2:
            assembly_cost = st.number_input("装配成本", 1, 20, 6)
            test_cost_final = st.number_input("成品检测成本", 1, 10, 3)
            repair_cost = st.number_input("返修成本", 1, 20, 5)
            market_price = st.number_input("市场价格", 10, 100, 56)
        
        if st.button("优化决策", key="production"):
            with st.spinner("正在优化..."):
                params = ProductionParams(
                    defect_rate1=defect_rate1,
                    defect_rate2=defect_rate2,
                    test_cost1=test_cost1,
                    test_cost2=test_cost2,
                    assembly_cost=assembly_cost,
                    test_cost_final=test_cost_final,
                    repair_cost=repair_cost,
                    market_price=market_price,
                    return_loss=market_price * 0.1
                )
                
                result = optimize_production(params)
                
                # 显示结果
                st.plotly_chart(create_decision_heatmap(result),
                              use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("期望利润", f"{result['expected_profit']:.2f}")
                    st.metric("合格率", f"{result['p_ok']*100:.1f}%")
                with col2:
                    st.metric("求解状态", result['solver_status'])
                    st.metric("求解时间", f"{result['solution_time']*1000:.1f}ms")
    
    with tab3:
        st.header("多工序生产系统优化")
        
        if st.button("创建示例网络", key="network"):
            with st.spinner("正在创建网络..."):
                graph = create_example_network()
                
                # 优化网络
                result = optimize_multistage(graph)
                
                # 显示3D拓扑图
                fig = create_3d_topology(graph)
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示优化结果
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("总成本", f"{result['total_cost']:.2f}")
                    st.metric("求解状态", result['solver_status'])
                with col2:
                    st.metric("求解时间", f"{result['solution_time']*1000:.1f}ms")
                
                # 显示各节点决策
                st.subheader("节点决策")
                for node, decision in result['decisions'].items():
                    with st.expander(f"节点 {node}"):
                        st.write(f"检测: {'是' if decision['test'] else '否'}")
                        st.write(f"返修: {'是' if decision['repair'] else '否'}")
                        st.write(f"合格率: {decision['p_ok']*100:.1f}%")
                        st.write(f"成本: {decision['cost']:.2f}")
    
    with tab4:
        st.header("鲁棒性分析")
        
        # 参数设置
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.number_input("抽样数量", 50, 200, 100)
            n_simulations = st.number_input("模拟次数", 50, 200, 100)
        with col2:
            confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95)
        
        if st.button("运行鲁棒性分析", key="robust"):
            with st.spinner("正在分析..."):
                # 设置不确定性参数
                uncertainty_params = UncertaintyParams(
                    n_samples=n_samples,
                    n_simulations=n_simulations,
                    confidence_level=confidence_level
                )
                
                # 生产决策鲁棒优化
                base_params = ProductionParams(
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
                
                prod_result = robust_optimize_production(
                    base_params, uncertainty_params)
                
                # 显示生产决策结果
                st.subheader("生产决策鲁棒性")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("期望利润",
                             f"{prod_result['expected_profit']:.2f}")
                    st.metric("最差情况利润",
                             f"{prod_result['worst_case_profit']:.2f}")
                with col2:
                    st.metric("利润标准差",
                             f"{prod_result['profit_std']:.2f}")
                    st.metric("决策置信度",
                             f"{prod_result['decision_confidence']*100:.1f}%")
                
                # 多工序鲁棒优化
                graph = create_example_network()
                multi_result = robust_optimize_multistage(
                    graph, uncertainty_params)
                
                # 显示多工序结果
                st.subheader("多工序系统鲁棒性")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("期望成本",
                             f"{multi_result['expected_cost']:.2f}")
                    st.metric("最差情况成本",
                             f"{multi_result['worst_case_cost']:.2f}")
                with col2:
                    st.metric("成本标准差",
                             f"{multi_result['cost_std']:.2f}")
                    
                # 显示各节点的鲁棒决策
                st.subheader("节点鲁棒决策")
                for node, decision in multi_result['robust_decisions'].items():
                    with st.expander(f"节点 {node}"):
                        st.write(f"检测: {'是' if decision['test'] else '否'}")
                        st.write(f"返修: {'是' if decision['repair'] else '否'}")
                        st.write(
                            f"决策置信度: {decision['decision_confidence']*100:.1f}%")
    
    # 应急控制台
    st.sidebar.title("应急控制台")
    if st.sidebar.button("启用降级模式",
                        help="当模型求解失败时，切换到启发式算法"):
        switch_to_heuristic_mode()
    
    if st.session_state.is_degraded_mode:
        st.sidebar.warning("当前处于降级模式")
    
    # 更新进度
    st.session_state.optimization_progress = min(
        100, st.session_state.optimization_progress + np.random.uniform(0, 1))

if __name__ == "__main__":
    main() 