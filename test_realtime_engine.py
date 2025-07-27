"""
实时决策引擎测试脚本
"""
import sys
sys.path.append('src')

from innovation.realtime_engine import RealtimeDecisionEngine, DecisionResult
import time
import numpy as np

def test_realtime_performance():
    """测试实时性能"""
    print("🚀 实时决策引擎性能测试")
    print("="*50)
    
    # 创建引擎
    engine = RealtimeDecisionEngine()
    
    # 设置回调函数
    high_priority_decisions = []
    
    def priority_callback(decision: DecisionResult):
        if decision.confidence > 0.85:
            high_priority_decisions.append(decision)
            print(f"⚡ 高优先级决策: {decision.action} | 置信度: {decision.confidence:.2f}")
    
    engine.add_decision_callback(priority_callback)
    
    # 运行测试
    print("启动实时数据流处理...")
    start_time = time.time()
    
    # 运行30秒的实时处理
    engine.process_live_data(duration_seconds=30)
    
    total_time = time.time() - start_time
    
    # 获取性能报告
    report = engine.get_performance_report()
    
    print(f"\n📊 性能测试结果:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"总运行时间: {total_time:.1f} 秒")
    print(f"处理数据量: {report['metrics']['processed_count']} 条")
    print(f"决策数量: {report['metrics']['decision_count']} 个")
    print(f"平均延迟: {report['avg_latency_ms']:.1f} ms")
    print(f"吞吐量: {report['metrics']['processed_count']/total_time:.1f} 条/秒")
    print(f"错误率: {report['error_rate']:.2%}")
    print(f"高优先级决策: {len(high_priority_decisions)} 个")
    print(f"数据库记录: {report['database_records']} 条")
    
    # 测试边缘计算部署
    print(f"\n🔧 边缘计算部署测试:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    edge_configs = engine.edge_computing()
    
    for platform, config in edge_configs.items():
        print(f"📱 {platform}:")
        print(f"   推理延迟: {config.get('inference_latency', 'N/A')}")
        print(f"   内存需求: {config.get('memory_requirement', 'N/A')}")
        print(f"   模型大小: {config.get('model_size', 'N/A')} bytes")
    
    # 验证实时性能指标
    print(f"\n✅ 实时性能验证:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    latency_ok = report['avg_latency_ms'] < 100  # 延迟小于100ms
    throughput_ok = report['metrics']['processed_count']/total_time > 1  # 吞吐量大于1条/秒
    error_rate_ok = report['error_rate'] < 0.01  # 错误率小于1%
    
    print(f"延迟要求 (<100ms): {'✅' if latency_ok else '❌'} {report['avg_latency_ms']:.1f}ms")
    print(f"吞吐量要求 (>1条/秒): {'✅' if throughput_ok else '❌'} {report['metrics']['processed_count']/total_time:.1f}条/秒")
    print(f"错误率要求 (<1%): {'✅' if error_rate_ok else '❌'} {report['error_rate']:.2%}")
    
    # 清理资源
    engine.stop()
    
    # 返回测试结果
    return {
        'latency_ms': report['avg_latency_ms'],
        'throughput_per_sec': report['metrics']['processed_count']/total_time,
        'error_rate': report['error_rate'],
        'total_decisions': report['metrics']['decision_count'],
        'high_priority_decisions': len(high_priority_decisions),
        'edge_deployment_ready': len(edge_configs) > 0
    }

def test_scalability():
    """测试可扩展性"""
    print(f"\n🔄 可扩展性测试")
    print("="*50)
    
    test_durations = [10, 20, 30]  # 不同的测试时长
    results = []
    
    for duration in test_durations:
        print(f"\n测试 {duration} 秒运行...")
        
        engine = RealtimeDecisionEngine()
        start_time = time.time()
        
        engine.process_live_data(duration_seconds=duration)
        
        actual_time = time.time() - start_time
        report = engine.get_performance_report()
        
        result = {
            'duration': duration,
            'actual_time': actual_time,
            'processed_count': report['metrics']['processed_count'],
            'avg_latency': report['avg_latency_ms'],
            'throughput': report['metrics']['processed_count'] / actual_time
        }
        
        results.append(result)
        engine.stop()
        
        print(f"  处理: {result['processed_count']} 条")
        print(f"  延迟: {result['avg_latency']:.1f} ms")
        print(f"  吞吐: {result['throughput']:.1f} 条/秒")
    
    # 分析扩展性
    print(f"\n📈 扩展性分析:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    for i, result in enumerate(results):
        if i > 0:
            prev_result = results[i-1]
            duration_ratio = result['duration'] / prev_result['duration']
            throughput_ratio = result['throughput'] / prev_result['throughput']
            scalability = throughput_ratio / duration_ratio
            
            print(f"时长 {prev_result['duration']}s → {result['duration']}s:")
            print(f"  时长倍数: {duration_ratio:.1f}x")
            print(f"  吞吐倍数: {throughput_ratio:.1f}x")
            print(f"  扩展性指标: {scalability:.2f} {'✅' if scalability > 0.8 else '❌'}")
    
    return results

def benchmark_decision_quality():
    """决策质量基准测试"""
    print(f"\n🎯 决策质量基准测试")
    print("="*50)
    
    engine = RealtimeDecisionEngine()
    
    # 收集决策数据
    decisions = []
    
    def collect_decisions(decision: DecisionResult):
        decisions.append(decision)
    
    engine.add_decision_callback(collect_decisions)
    
    # 运行决策收集
    engine.process_live_data(duration_seconds=20)
    
    # 分析决策质量
    if decisions:
        confidences = [d.confidence for d in decisions]
        benefits = [d.expected_benefit for d in decisions]
        
        avg_confidence = np.mean(confidences)
        avg_benefit = np.mean(benefits)
        high_confidence_ratio = sum(1 for c in confidences if c > 0.8) / len(confidences)
        
        decision_types = {}
        for d in decisions:
            decision_types[d.decision_type] = decision_types.get(d.decision_type, 0) + 1
        
        print(f"决策数量: {len(decisions)}")
        print(f"平均置信度: {avg_confidence:.3f}")
        print(f"平均预期收益: {avg_benefit:.1f}")
        print(f"高置信度比例: {high_confidence_ratio:.1%}")
        print(f"决策类型分布:")
        for dtype, count in decision_types.items():
            print(f"  {dtype}: {count} ({count/len(decisions):.1%})")
    
    engine.stop()
    
    return {
        'total_decisions': len(decisions),
        'avg_confidence': avg_confidence if decisions else 0,
        'avg_benefit': avg_benefit if decisions else 0,
        'high_confidence_ratio': high_confidence_ratio if decisions else 0
    }

def generate_test_report():
    """生成测试报告"""
    print("🔬 实时决策引擎综合测试报告")
    print("="*60)
    
    # 运行所有测试
    print("正在运行性能测试...")
    perf_results = test_realtime_performance()
    
    print("正在运行可扩展性测试...")
    scale_results = test_scalability()
    
    print("正在运行决策质量测试...")
    quality_results = benchmark_decision_quality()
    
    # 生成报告
    print(f"\n📋 综合测试报告")
    print("="*60)
    
    print(f"\n🚀 性能指标:")
    print(f"  平均延迟: {perf_results['latency_ms']:.1f} ms")
    print(f"  数据吞吐: {perf_results['throughput_per_sec']:.1f} 条/秒")
    print(f"  错误率: {perf_results['error_rate']:.2%}")
    print(f"  总决策数: {perf_results['total_decisions']}")
    
    print(f"\n📈 扩展性指标:")
    if len(scale_results) >= 2:
        last_result = scale_results[-1]
        first_result = scale_results[0]
        scale_efficiency = (last_result['throughput'] / first_result['throughput']) / (last_result['duration'] / first_result['duration'])
        print(f"  扩展效率: {scale_efficiency:.2f}")
        print(f"  最大吞吐: {max(r['throughput'] for r in scale_results):.1f} 条/秒")
    
    print(f"\n🎯 决策质量:")
    print(f"  平均置信度: {quality_results['avg_confidence']:.3f}")
    print(f"  平均预期收益: {quality_results['avg_benefit']:.1f}")
    print(f"  高置信度比例: {quality_results['high_confidence_ratio']:.1%}")
    
    print(f"\n🔧 边缘部署:")
    print(f"  部署就绪: {'✅' if perf_results['edge_deployment_ready'] else '❌'}")
    
    print(f"\n🏆 总体评估:")
    
    # 计算综合评分
    performance_score = min(100, (100 - perf_results['latency_ms']) + perf_results['throughput_per_sec'] * 10)
    quality_score = quality_results['avg_confidence'] * 100
    reliability_score = (1 - perf_results['error_rate']) * 100
    
    overall_score = (performance_score + quality_score + reliability_score) / 3
    
    print(f"  性能评分: {performance_score:.1f}/100")
    print(f"  质量评分: {quality_score:.1f}/100")
    print(f"  可靠性评分: {reliability_score:.1f}/100")
    print(f"  综合评分: {overall_score:.1f}/100")
    
    if overall_score >= 90:
        print(f"  等级: 🥇 优秀")
    elif overall_score >= 80:
        print(f"  等级: 🥈 良好")
    elif overall_score >= 70:
        print(f"  等级: 🥉 合格")
    else:
        print(f"  等级: ❌ 需要改进")
    
    print(f"\n💡 建议:")
    if perf_results['latency_ms'] > 50:
        print(f"  • 优化算法以降低延迟")
    if perf_results['error_rate'] > 0.005:
        print(f"  • 增强错误处理机制")
    if quality_results['high_confidence_ratio'] < 0.7:
        print(f"  • 改进决策模型准确性")
    
    print(f"\n✅ 实时决策引擎测试完成！")
    
    return {
        'performance': perf_results,
        'scalability': scale_results,
        'quality': quality_results,
        'overall_score': overall_score
    }

if __name__ == '__main__':
    # 运行综合测试
    test_results = generate_test_report()
    
    # 保存测试结果
    import json
    with open('output/realtime_engine_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False) 