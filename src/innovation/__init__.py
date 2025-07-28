"""
创新算法模块
"""

# 延迟导入，避免启动时的依赖问题
def _lazy_import():
    """延迟导入模块"""
    global NationalAwardEnhancer, ExascaleOptimizer, ExascaleParams
    
    try:
        from .national_champion import NationalAwardEnhancer
    except ImportError as e:
        print(f"警告: NationalAwardEnhancer导入失败: {e}")
        NationalAwardEnhancer = None
    
    try:
        from .exascale_optimizer import ExascaleOptimizer, ExascaleParams
    except ImportError as e:
        print(f"警告: ExascaleOptimizer导入失败: {e}")
        ExascaleOptimizer = None
        ExascaleParams = None

# 初始化全局变量
NationalAwardEnhancer = None
ExascaleOptimizer = None
ExascaleParams = None

def __getattr__(name):
    """按需导入"""
    if name in ['NationalAwardEnhancer', 'ExascaleOptimizer', 'ExascaleParams']:
        if globals()[name] is None:
            _lazy_import()
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'NationalAwardEnhancer',
    'ExascaleOptimizer', 
    'ExascaleParams'
]

__version__ = '0.1.0' 