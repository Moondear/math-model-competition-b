#!/usr/bin/env python3
"""
增强版AI答辩教练系统
集成智能评分、弱点分析、压力训练等功能
"""

import random
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class Question:
    """问题数据结构"""
    id: str
    category: str
    difficulty: int  # 1-5
    question: str
    keywords: List[str]
    standard_answer: str
    scoring_criteria: Dict[str, int]

@dataclass
class Answer:
    """答案数据结构"""
    content: str
    score: int
    feedback: str
    suggestions: List[str]
    keywords_found: List[str]
    missing_keywords: List[str]

class QuestionBank:
    """智能问题库"""
    
    def __init__(self):
        self.questions = self._initialize_questions()
    
    def _initialize_questions(self) -> List[Question]:
        """初始化问题库"""
        return [
            Question(
                id="tech_001",
                category="技术原理",
                difficulty=3,
                question="请详细阐述您的数学模型的核心思想和理论基础",
                keywords=["数学模型", "优化", "算法", "理论", "假设", "约束", "目标函数"],
                standard_answer="我们建立了一个基于整数规划的多目标优化模型，核心思想是在满足生产约束的前提下，最大化期望利润并最小化风险。",
                scoring_criteria={"模型描述": 30, "理论基础": 25, "逻辑清晰": 25, "表达准确": 20}
            ),
            Question(
                id="tech_002", 
                category="算法复杂度",
                difficulty=4,
                question="您的算法时间复杂度是多少？如何优化计算效率？",
                keywords=["时间复杂度", "空间复杂度", "优化", "算法", "效率", "并行", "加速"],
                standard_answer="主算法时间复杂度为O(n log n)，通过并行计算和启发式优化，实际处理千万变量仅需1.1秒。",
                scoring_criteria={"复杂度分析": 35, "优化方法": 30, "实际性能": 25, "表达清晰": 10}
            ),
            Question(
                id="inno_001",
                category="创新亮点", 
                difficulty=3,
                question="与传统方法相比，您的主要创新点在哪里？",
                keywords=["创新", "改进", "优势", "突破", "新颖", "差异", "贡献"],
                standard_answer="主要创新包括：量子启发优化算法、联邦学习预测、区块链记录和VR/AR展示系统。",
                scoring_criteria={"创新识别": 30, "技术深度": 30, "应用价值": 25, "表达完整": 15}
            ),
            Question(
                id="valid_001",
                category="验证分析",
                difficulty=3,
                question="如何验证模型的准确性和可靠性？",
                keywords=["验证", "测试", "准确性", "可靠性", "实验", "对比", "评估"],
                standard_answer="通过100次压力测试验证稳定性，与传统方法对比验证准确性，实际案例验证实用性。",
                scoring_criteria={"验证方法": 35, "实验设计": 25, "结果分析": 25, "可信度": 15}
            ),
            Question(
                id="app_001",
                category="应用场景",
                difficulty=2,
                question="您的模型在实际应用中有什么局限性？",
                keywords=["局限性", "限制", "适用范围", "条件", "假设", "改进"],
                standard_answer="主要局限包括：数据质量要求较高、计算资源需求、模型参数需要调优等。",
                scoring_criteria={"问题识别": 30, "分析深度": 30, "改进方案": 25, "诚实度": 15}
            ),
            Question(
                id="impl_001",
                category="实现细节",
                difficulty=4,
                question="在实际部署中遇到了哪些技术挑战？如何解决？",
                keywords=["部署", "挑战", "解决方案", "技术难点", "优化", "改进"],
                standard_answer="主要挑战包括OR-Tools兼容性、大规模计算和实时响应，通过备用算法和并行优化解决。",
                scoring_criteria={"挑战识别": 25, "解决方案": 35, "技术深度": 25, "实施效果": 15}
            ),
            Question(
                id="perf_001",
                category="性能分析",
                difficulty=3,
                question="您如何评估算法的性能表现？有哪些关键指标？",
                keywords=["性能", "指标", "评估", "基准", "测试", "对比", "优化"],
                standard_answer="关键指标包括：处理速度1.1秒、内存使用0.6MB、并发能力1097请求/秒、准确率92%。",
                scoring_criteria={"指标选择": 30, "测试方法": 30, "结果分析": 25, "对比评估": 15}
            )
        ]
    
    def get_question_by_category(self, category: str) -> Optional[Question]:
        """按类别获取问题"""
        questions = [q for q in self.questions if q.category == category]
        return random.choice(questions) if questions else None
    
    def get_random_question(self, difficulty_range: Tuple[int, int] = (1, 5)) -> Question:
        """获取随机问题"""
        min_diff, max_diff = difficulty_range
        suitable_questions = [q for q in self.questions 
                            if min_diff <= q.difficulty <= max_diff]
        return random.choice(suitable_questions)

class IntelligentScorer:
    """智能评分系统"""
    
    def __init__(self):
        self.stop_words = {'的', '是', '在', '有', '和', '与', '等', '以及', '通过'}
    
    def score_answer(self, question: Question, answer: str) -> Answer:
        """智能评分答案"""
        answer_words = set(self._extract_keywords(answer))
        question_keywords = set(question.keywords)
        
        # 关键词匹配分析
        keywords_found = list(answer_words.intersection(question_keywords))
        missing_keywords = list(question_keywords - answer_words)
        
        # 基础分数计算
        keyword_coverage = len(keywords_found) / len(question.keywords)
        base_score = int(keyword_coverage * 70)  # 关键词覆盖度基础分
        
        # 长度奖励
        length_bonus = min(len(answer) // 50, 10)  # 每50字符奖励1分，最多10分
        
        # 专业词汇奖励
        technical_words = ['优化', '算法', '模型', '分析', '设计', '实现', '验证', '测试']
        tech_bonus = len(set(self._extract_keywords(answer)).intersection(technical_words)) * 2
        
        # 计算最终得分
        total_score = min(base_score + length_bonus + tech_bonus, 95)
        total_score = max(total_score, 40)  # 最低40分
        
        # 生成反馈和建议
        feedback, suggestions = self._generate_feedback(
            total_score, keywords_found, missing_keywords, question
        )
        
        return Answer(
            content=answer,
            score=total_score,
            feedback=feedback,
            suggestions=suggestions,
            keywords_found=keywords_found,
            missing_keywords=missing_keywords
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的中文分词（实际应用中可以使用jieba等工具）
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text)
        return [w for w in words if w not in self.stop_words and len(w) > 1]
    
    def _generate_feedback(self, score: int, found: List[str], missing: List[str], 
                          question: Question) -> Tuple[str, List[str]]:
        """生成反馈和建议"""
        feedback = f"得分: {score}/100"
        suggestions = []
        
        if score >= 85:
            feedback += " - 优秀回答！"
            suggestions.append("回答质量很高，继续保持")
        elif score >= 70:
            feedback += " - 良好回答"
            suggestions.append("回答基本到位，可以增加更多细节")
        elif score >= 60:
            feedback += " - 一般回答"
            suggestions.append("回答需要更多技术细节支撑")
        else:
            feedback += " - 需要改进"
            suggestions.append("回答缺乏核心要点，需要重新组织")
        
        if missing:
            suggestions.append(f"建议补充关键词: {', '.join(missing[:3])}")
        
        if len(found) > 0:
            suggestions.append(f"已涵盖关键概念: {', '.join(found[:3])}")
        
        return feedback, suggestions

class WeaknessAnalyzer:
    """弱点分析器"""
    
    def __init__(self):
        self.category_performance = defaultdict(list)
        self.common_mistakes = defaultdict(int)
    
    def record_performance(self, question: Question, answer: Answer):
        """记录表现"""
        self.category_performance[question.category].append(answer.score)
        
        if answer.score < 70:
            self.common_mistakes[question.category] += 1
    
    def analyze_weaknesses(self) -> Dict:
        """分析弱点"""
        analysis = {
            'category_scores': {},
            'weak_categories': [],
            'improvement_suggestions': []
        }
        
        for category, scores in self.category_performance.items():
            avg_score = np.mean(scores)
            analysis['category_scores'][category] = avg_score
            
            if avg_score < 70:
                analysis['weak_categories'].append(category)
        
        # 生成改进建议
        for weak_cat in analysis['weak_categories']:
            suggestion = self._get_improvement_suggestion(weak_cat)
            analysis['improvement_suggestions'].append(suggestion)
        
        return analysis
    
    def _get_improvement_suggestion(self, category: str) -> str:
        """获取改进建议"""
        suggestions = {
            '技术原理': '建议深入学习数学建模理论基础，多练习模型构建',
            '算法复杂度': '需要加强算法分析能力，学习时空复杂度计算',
            '创新亮点': '注意总结项目创新点，准备具体的对比分析',
            '验证分析': '学习实验设计方法，准备详细的验证方案',
            '应用场景': '多了解实际应用案例，思考模型的适用范围',
            '实现细节': '加强编程实践，熟悉常见技术难点及解决方案',
            '性能分析': '学习性能评估方法，准备关键性能指标'
        }
        return suggestions.get(category, '加强该领域的理论学习和实践')

class PressureTrainer:
    """压力训练器"""
    
    def __init__(self):
        self.stress_scenarios = [
            "投影仪突然黑屏，请继续您的讲解",
            "网络中断，无法展示在线演示，请口述您的核心算法",
            "专家打断：这个算法我不理解，请用最简单的话解释",
            "时间不够了，请在30秒内总结您的主要贡献",
            "专家质疑：这个结果看起来不可信，您如何解释？"
        ]
    
    def generate_stress_scenario(self) -> str:
        """生成压力场景"""
        return random.choice(self.stress_scenarios)
    
    def evaluate_stress_response(self, response: str, scenario: str) -> Dict:
        """评估压力响应"""
        # 简化的压力响应评估
        response_length = len(response)
        calmness_score = min(response_length // 30, 50)  # 基于回答长度评估冷静度
        
        # 检查是否包含关键应对词汇
        stress_keywords = ['理解', '解释', '可以', '没问题', '简单来说', '总结']
        stress_handling = sum(1 for word in stress_keywords if word in response) * 10
        
        total_score = min(calmness_score + stress_handling, 100)
        
        return {
            'stress_score': total_score,
            'calmness': calmness_score,
            'adaptability': stress_handling,
            'feedback': f"压力应对得分: {total_score}/100"
        }

class EnhancedDefenseCoach:
    """增强版AI答辩教练"""
    
    def __init__(self):
        self.question_bank = QuestionBank()
        self.scorer = IntelligentScorer()
        self.weakness_analyzer = WeaknessAnalyzer()
        self.pressure_trainer = PressureTrainer()
        
        self.session_data = {
            'questions_asked': 0,
            'total_score': 0,
            'category_performance': defaultdict(list),
            'training_history': [],
            'start_time': datetime.now()
        }
    
    def start_standard_training(self, rounds: int = 10) -> Dict:
        """标准训练模式"""
        print(f"🎯 开始标准答辩训练 ({rounds}轮)")
        print("=" * 60)
        
        results = []
        
        for i in range(rounds):
            print(f"\n【第{i+1}轮训练】")
            
            # 获取问题
            question = self.question_bank.get_random_question()
            print(f"问题类别: {question.category}")
            print(f"问题难度: {'⭐' * question.difficulty}")
            print(f"问题: {question.question}")
            
            # 模拟回答（实际使用中可以等待用户输入）
            simulated_answer = self._generate_simulated_answer(question)
            print(f"回答: {simulated_answer}")
            
            # 评分
            answer_result = self.scorer.score_answer(question, simulated_answer)
            
            # 记录结果
            self.weakness_analyzer.record_performance(question, answer_result)
            self._update_session_data(question, answer_result)
            
            print(f"得分: {answer_result.score}/100")
            print(f"反馈: {answer_result.feedback}")
            if answer_result.suggestions:
                print(f"建议: {'; '.join(answer_result.suggestions)}")
            
            results.append({
                'round': i + 1,
                'question': question.question,
                'category': question.category,
                'score': answer_result.score,
                'feedback': answer_result.feedback
            })
            
            time.sleep(0.1)  # 短暂停顿
        
        return self._generate_training_report(results)
    
    def start_pressure_training(self, rounds: int = 5) -> Dict:
        """压力训练模式"""
        print(f"\n💥 开始压力训练模式 ({rounds}轮)")
        print("=" * 60)
        
        pressure_results = []
        
        for i in range(rounds):
            print(f"\n【压力测试 {i+1}】")
            
            # 生成压力场景
            scenario = self.pressure_trainer.generate_stress_scenario()
            print(f"🚨 突发情况: {scenario}")
            
            # 获取问题
            question = self.question_bank.get_random_question((3, 5))  # 较高难度
            print(f"问题: {question.question}")
            
            # 模拟压力下的回答
            stress_answer = self._generate_stress_answer(question, scenario)
            print(f"回答: {stress_answer}")
            
            # 压力响应评估
            stress_eval = self.pressure_trainer.evaluate_stress_response(stress_answer, scenario)
            
            # 常规评分
            answer_eval = self.scorer.score_answer(question, stress_answer)
            
            print(f"常规得分: {answer_eval.score}/100")
            print(f"压力应对: {stress_eval['stress_score']}/100")
            
            pressure_results.append({
                'round': i + 1,
                'scenario': scenario,
                'question': question.question,
                'regular_score': answer_eval.score,
                'stress_score': stress_eval['stress_score']
            })
        
        return pressure_results
    
    def get_weakness_analysis(self) -> Dict:
        """获取弱点分析"""
        return self.weakness_analyzer.analyze_weaknesses()
    
    def generate_improvement_plan(self) -> Dict:
        """生成改进计划"""
        weakness_analysis = self.get_weakness_analysis()
        
        plan = {
            'total_training_time': 180,  # 3小时
            'modules': [],
            'priority_areas': weakness_analysis['weak_categories']
        }
        
        # 基于弱点生成训练模块
        for category in weakness_analysis['weak_categories']:
            module = {
                'name': f"{category}强化训练",
                'duration': 30,
                'activities': [
                    f"深入学习{category}相关理论",
                    f"练习{category}类问题回答",
                    f"准备{category}标准话术"
                ]
            }
            plan['modules'].append(module)
        
        return plan
    
    def _generate_simulated_answer(self, question: Question) -> str:
        """生成模拟回答"""
        # 基于问题关键词生成合理的模拟回答
        answer_templates = {
            '技术原理': "我们的模型基于{keywords}，通过{method}实现{goal}。",
            '算法复杂度': "算法时间复杂度为O(n log n)，通过{optimization}优化性能。",
            '创新亮点': "主要创新包括{innovation1}、{innovation2}和{innovation3}。",
            '验证分析': "我们通过{validation_method}验证，包括{test1}和{test2}。",
            '应用场景': "适用于{scenario}，但在{limitation}方面存在限制。",
            '实现细节': "实现中遇到{challenge}，通过{solution}解决。",
            '性能分析': "关键性能指标包括{metric1}、{metric2}和{metric3}。"
        }
        
        template = answer_templates.get(question.category, "根据{keywords}分析，我们{action}。")
        
        # 简单的模板填充
        keywords_str = "、".join(question.keywords[:3])
        filled_answer = template.format(
            keywords=keywords_str,
            method="优化算法",
            goal="最佳性能",
            optimization="并行计算",
            innovation1="量子启发优化",
            innovation2="联邦学习",
            innovation3="区块链记录",
            validation_method="多重验证",
            test1="压力测试",
            test2="对比实验",
            scenario="大规模生产优化",
            limitation="数据质量",
            challenge="OR-Tools兼容性",
            solution="备用算法",
            metric1="处理速度",
            metric2="内存使用",
            metric3="准确率",
            action="进行了深入研究"
        )
        
        return filled_answer
    
    def _generate_stress_answer(self, question: Question, scenario: str) -> str:
        """生成压力情况下的回答"""
        # 压力下的回答通常更简短、更直接
        stress_answer = self._generate_simulated_answer(question)
        
        # 添加应对压力的话术
        stress_responses = [
            "没问题，我来简单解释一下。",
            "理解，让我用更直观的方式说明。",
            "好的，我总结一下核心要点。"
        ]
        
        response_prefix = random.choice(stress_responses)
        return f"{response_prefix} {stress_answer[:100]}..."  # 缩短回答
    
    def _update_session_data(self, question: Question, answer: Answer):
        """更新会话数据"""
        self.session_data['questions_asked'] += 1
        self.session_data['total_score'] += answer.score
        self.session_data['category_performance'][question.category].append(answer.score)
        
        self.session_data['training_history'].append({
            'timestamp': datetime.now(),
            'question_id': question.id,
            'category': question.category,
            'score': answer.score
        })
    
    def _generate_training_report(self, results: List[Dict]) -> Dict:
        """生成训练报告"""
        if not results:
            return {}
        
        scores = [r['score'] for r in results]
        avg_score = np.mean(scores)
        
        report = {
            'summary': {
                'total_rounds': len(results),
                'average_score': avg_score,
                'highest_score': max(scores),
                'lowest_score': min(scores),
                'improvement_rate': (scores[-1] - scores[0]) / scores[0] * 100 if scores[0] > 0 else 0
            },
            'category_performance': self.session_data['category_performance'],
            'detailed_results': results,
            'overall_rating': self._get_performance_rating(avg_score),
            'next_steps': self._get_next_steps(avg_score)
        }
        
        return report
    
    def _get_performance_rating(self, avg_score: float) -> str:
        """获取表现评级"""
        if avg_score >= 85:
            return "🥇 优秀"
        elif avg_score >= 75:
            return "🥈 良好"
        elif avg_score >= 65:
            return "🥉 合格"
        else:
            return "📈 需要提升"
    
    def _get_next_steps(self, avg_score: float) -> List[str]:
        """获取下一步建议"""
        if avg_score >= 85:
            return ["保持当前水平", "准备更高难度问题", "增加压力训练"]
        elif avg_score >= 75:
            return ["继续练习薄弱环节", "增加技术细节", "提升表达流畅度"]
        elif avg_score >= 65:
            return ["加强基础理论学习", "多练习标准答案", "改善回答结构"]
        else:
            return ["重新梳理项目核心", "加强基础训练", "寻求专业指导"]

# 兼容性接口
class DefenseCoach(EnhancedDefenseCoach):
    """保持向后兼容的接口"""
    
    def ask_question(self) -> str:
        """获取问题"""
        question = self.question_bank.get_random_question()
        return question.question
    
    def evaluate_answer(self, answer: str) -> Dict:
        """评估答案"""
        # 使用最后一个问题进行评估（简化版）
        question = self.question_bank.get_random_question()
        result = self.scorer.score_answer(question, answer)
        
        return {
            'score': result.score,
            'feedback': result.feedback,
            'suggestions': result.suggestions
        }
    
    def get_session_summary(self) -> Dict:
        """获取会话总结"""
        if self.session_data['questions_asked'] == 0:
            return {
                'questions_asked': 0,
                'correct_answers': 0,
                'average_score': 0,
                'success_rate': 0
            }
        
        avg_score = self.session_data['total_score'] / self.session_data['questions_asked']
        success_rate = sum(1 for scores in self.session_data['category_performance'].values() 
                          for score in scores if score >= 70) / self.session_data['questions_asked'] * 100
        
        return {
            'questions_asked': self.session_data['questions_asked'],
            'correct_answers': int(success_rate * self.session_data['questions_asked'] / 100),
            'average_score': avg_score,
            'success_rate': success_rate
        }

class DefenseTrainingSystem:
    """答辩训练系统"""
    
    def __init__(self):
        self.coach = EnhancedDefenseCoach()
        self.training_sessions = []
    
    def start_training_session(self, rounds: int = 10) -> Dict:
        """开始训练会话"""
        result = self.coach.start_standard_training(rounds)
        self.training_sessions.append(result)
        return result
    
    def pressure_training(self) -> Dict:
        """压力训练"""
        return self.coach.start_pressure_training(5) 