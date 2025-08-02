#!/usr/bin/env python3
"""测试量子能量景观图生成"""

import numpy as np
import plotly.graph_objects as go

def test_quantum_plot():
    """测试量子能量景观图生成"""
    print("正在测试量子能量景观图生成...")
    
    # 创建3D参数空间网格
    x = np.linspace(-4, 4, 50)  # 减少网格点数以提高性能
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    
    # 定义复杂的多峰能量函数
    Z = (
        # 主要势能井
        2 * np.exp(-((X-1)**2 + (Y-1)**2)/2) +
        1.5 * np.exp(-((X+1.5)**2 + (Y+0.5)**2)/1.5) +
        
        # 能量壁垒
        0.8 * np.exp(-((X)**2 + (Y+2)**2)/3) +
        
        # 波动项（量子特性）
        0.3 * np.sin(1.5*X) * np.cos(1.5*Y) +
        
        # 全局最优（深势阱）
        -3 * np.exp(-((X+0.5)**2 + (Y-1.5)**2)/0.8)
    )
    
    # 添加一些噪声模拟量子涨落
    np.random.seed(42)
    Z += 0.1 * np.random.normal(0, 1, Z.shape)
    
    # 创建3D图
    fig = go.Figure()
    
    # 添加能量表面
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.8,
        name='量子势能面'
    ))
    
    # 添加标题和布局
    fig.update_layout(
        title='量子启发优化算法 - 3D能量景观图',
        scene=dict(
            xaxis=dict(title='参数空间 X'),
            yaxis=dict(title='参数空间 Y'),
            zaxis=dict(title='能量值')
        ),
        width=1000,
        height=800
    )
    
    # 保存文件
    try:
        fig.write_html("output/quantum_energy_landscape_test.html")
        print("✅ 量子能量景观图测试成功!")
        return True
    except Exception as e:
        print(f"❌ 生成失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_quantum_plot()