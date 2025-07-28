#!/usr/bin/env python3
"""
简化版AI答辩教练系统
"""

import random
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

class DefenseCoach:
    """AI答辩教练"""
    
    def __init__(self):
        self.questions = [
            "请解释您的数学模型的核心思想",
            "算法的时间复杂度是多少？",
            "如何验证模型的准确性？",
            "与传统方法相比，您的创新点在哪里？",
            "模型在实际应用中的局限性是什么？"
        ]
        self.session_data = {
            'questions_asked': 0,
            'correct_answers': 0,
            'total_score': 0
        }
    
    def ask_question(self) -> str:
        """随机提出一个问题"""
        question = random.choice(self.questions)
        self.session_data['questions_asked'] += 1
        return question
    
    def evaluate_answer(self, answer: str) -> Dict:
        """评估答案质量"""
        # 简化评分逻辑
        score = random.randint(60, 85)  # 模拟评分
        
        if score >= 70:
            self.session_data['correct_answers'] += 1
        
        self.session_data['total_score'] += score
        
        return {
            'score': score,
            'feedback': f"答案得分: {score}/100",
            'suggestions': ["可以补充更多技术细节", "逻辑表达可以更清晰"]
        }
    
    def get_session_summary(self) -> Dict:
        """获取训练总结"""
        avg_score = (self.session_data['total_score'] / 
                    max(self.session_data['questions_asked'], 1))
        
        return {
            'questions_asked': self.session_data['questions_asked'],
            'correct_answers': self.session_data['correct_answers'],
            'average_score': avg_score,
            'success_rate': (self.session_data['correct_answers'] / 
                           max(self.session_data['questions_asked'], 1)) * 100
        }

class DefenseTrainingSystem:
    """答辩训练系统"""
    
    def __init__(self):
        self.coach = DefenseCoach()
        self.training_sessions = []
    
    def start_training_session(self, rounds: int = 10) -> Dict:
        """开始训练会话"""
        print(f"🎯 开始AI答辩特训 ({rounds}轮)")
        print("="*50)
        
        session_start = datetime.now()
        
        for i in range(rounds):
            print(f"\n第{i+1}轮训练:")
            question = self.coach.ask_question()
            print(f"问题: {question}")
            
            # 模拟回答（实际应用中这里会等待用户输入）
            simulated_answer = "这是一个模拟回答..."
            time.sleep(0.1)  # 模拟思考时间
            
            evaluation = self.coach.evaluate_answer(simulated_answer)
            print(f"评分: {evaluation['score']}/100")
            print(f"反馈: {evaluation['feedback']}")
        
        session_summary = self.coach.get_session_summary()
        session_end = datetime.now()
        
        # 保存训练记录
        training_record = {
            'timestamp': session_start.isoformat(),
            'duration': str(session_end - session_start),
            'rounds': rounds,
            'summary': session_summary
        }
        self.training_sessions.append(training_record)
        
        print(f"\n🏆 训练完成!")
        print(f"总计问题: {session_summary['questions_asked']}")
        print(f"正确回答: {session_summary['correct_answers']}")
        print(f"平均得分: {session_summary['average_score']:.1f}/100")
        print(f"成功率: {session_summary['success_rate']:.1f}%")
        
        return training_record
    
    def pressure_training(self) -> Dict:
        """压力训练模式"""
        print("💥 启动压力训练模式")
        print("模拟设备故障、时间压力等情况...")
        
        # 模拟压力测试
        stress_factors = [
            "⚡ 模拟投影仪故障",
            "⏰ 模拟时间不足压力", 
            "🔥 模拟专家连续追问",
            "💻 模拟电脑死机"
        ]
        
        for factor in stress_factors:
            print(f"  {factor}...")
            time.sleep(0.2)
        
        # 评估抗压能力
        stress_score = random.randint(85, 95)
        
        print(f"✅ 压力训练完成!")
        print(f"抗压能力评分: {stress_score}/100")
        
        return {
            'stress_score': stress_score,
            'factors_tested': len(stress_factors),
            'recommendation': "抗压能力优秀，建议继续保持"
        }

# 兼容性包装
class AIDefenseCoach(DefenseCoach):
    """兼容性包装类"""
    pass 