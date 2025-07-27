"""
独立脚本：生成AI答辩问答知识库Word文档
"""

from src.defense.ai_defense_coach import QuestionKnowledgeBase

def main():
    """主函数：生成Word格式知识库"""
    print("📚 正在生成AI答辩问答知识库Word文档...")
    
    # 创建知识库实例
    knowledge_base = QuestionKnowledgeBase()
    
    # 导出到Word文档
    result = knowledge_base.export_to_word("AI答辩问答知识库.docx")
    print(result)
    
    # 也生成一个备用的文本版本
    text_result = knowledge_base._export_to_text("AI答辩问答知识库_完整版.txt")
    print(text_result)
    
    print("\n📋 知识库文档说明:")
    print("📄 Word版本: AI答辩问答知识库.docx (推荐，格式精美)")
    print("📄 文本版本: AI答辩问答知识库_完整版.txt (备用版本)")
    print()
    print("📊 知识库内容概览:")
    print(f"   总问题数: {len(knowledge_base.questions)}题")
    
    # 按类别统计
    categories = {
        'technical_details': '技术细节类',
        'innovation_points': '创新点类',
        'practical_application': '实际应用类',
        'theoretical_foundation': '理论基础类'
    }
    
    for category_id, category_name in categories.items():
        count = len([q for q in knowledge_base.questions.values() if q.category == category_id])
        print(f"   {category_name}: {count}题")
    
    print("\n🎯 使用建议:")
    print("1. 优先使用Word版本，格式更清晰美观")
    print("2. 每个问题包含：难度、框架、标准答案、评分标准")
    print("3. 可以打印出来作为答辩准备材料")
    print("4. 配合AI答辩系统使用效果最佳")

if __name__ == "__main__":
    main() 