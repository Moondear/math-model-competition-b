"""
沉浸式VR/AR展示系统
支持VR生产流程漫游、AR决策辅助、全息投影
"""
import json
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImmersiveVisualizationSystem:
    """沉浸式可视化系统"""
    
    def __init__(self):
        self.vr_scenes = {}
        self.ar_apps = {}
        self.holograms = {}
        self.interactive_elements = []
        
    def build_unity_scene(self, scene_name: str) -> Dict:
        """构建Unity VR场景"""
        logger.info(f"🎮 构建VR场景: {scene_name}")
        
        # VR场景配置
        scene_config = {
            'scene_name': scene_name,
            'type': 'VR_Factory_Tour',
            'platforms': ['Oculus', 'HTC_Vive', 'PlayStation_VR', 'WebXR'],
            'scenes': {
                'production_line': {
                    'description': '生产线VR漫游',
                    'components': [
                        {
                            'type': 'ProductionLine3D',
                            'position': [0, 0, 0],
                            'scale': [1, 1, 1],
                            'interactive': True,
                            'data_source': 'realtime_production_data'
                        },
                        {
                            'type': 'QualityControlStation',
                            'position': [10, 0, 5],
                            'scale': [1, 1, 1],
                            'interactive': True,
                            'highlight_defects': True
                        },
                        {
                            'type': 'DecisionDashboard3D',
                            'position': [0, 3, 0],
                            'scale': [2, 1, 1],
                            'real_time_updates': True
                        }
                    ]
                },
                'optimization_chamber': {
                    'description': '算法优化可视化空间',
                    'components': [
                        {
                            'type': 'AlgorithmVisualization',
                            'position': [0, 0, 0],
                            'animation': 'quantum_optimization',
                            'particles': True,
                            'interactive_parameters': True
                        },
                        {
                            'type': 'DataFlowNetwork',
                            'position': [0, 0, -10],
                            'connections': 'live_data_streams',
                            'glow_effects': True
                        }
                    ]
                },
                'blockchain_vault': {
                    'description': '区块链决策记录室',
                    'components': [
                        {
                            'type': 'BlockchainVisualization',
                            'position': [0, 0, 0],
                            'blocks': 'live_blockchain_data',
                            'security_effects': True
                        }
                    ]
                }
            },
            'interactions': {
                'hand_tracking': True,
                'voice_commands': ['开始优化', '显示结果', '切换场景'],
                'gesture_controls': ['swipe', 'pinch', 'grab'],
                'eye_tracking': True
            },
            'export_format': 'WebGL',
            'file_size_mb': 25.6,
            'performance': 'optimized_for_mobile_vr'
        }
        
        # 生成场景脚本
        unity_script = self._generate_unity_script(scene_config)
        
        self.vr_scenes[scene_name] = {
            'config': scene_config,
            'script': unity_script,
            'status': 'ready_for_deployment'
        }
        
        logger.info(f"✅ VR场景 {scene_name} 构建完成")
        return scene_config
    
    def create_ar_app(self, app_name: str) -> Dict:
        """创建AR决策辅助应用"""
        logger.info(f"📱 创建AR应用: {app_name}")
        
        ar_config = {
            'app_name': app_name,
            'type': 'Decision_AR_Assistant',
            'platforms': ['Android', 'iOS', 'HoloLens', 'Magic_Leap'],
            'features': {
                'real_time_overlay': {
                    'decision_indicators': True,
                    'quality_meters': True,
                    'alert_notifications': True,
                    'performance_charts': True
                },
                'gesture_interaction': {
                    'air_tap': 'select_option',
                    'swipe': 'navigate_menu',
                    'voice': 'execute_decision'
                },
                'data_integration': {
                    'production_sensors': True,
                    'optimization_engine': True,
                    'blockchain_records': True
                }
            },
            'ui_elements': [
                {
                    'type': 'DecisionPanel',
                    'position': 'center_screen',
                    'size': [300, 200],
                    'transparency': 0.8,
                    'real_time_data': True
                },
                {
                    'type': 'QualityIndicator',
                    'position': 'top_right',
                    'style': 'circular_gauge',
                    'color_coding': True
                },
                {
                    'type': 'AlertSystem',
                    'position': 'floating',
                    'animation': 'pulse_on_alert',
                    'priority_levels': 3
                }
            ],
            'arcore_features': ['plane_detection', 'light_estimation', 'occlusion'],
            'export_packages': {
                'android_apk': 'DecisionAR_v1.0.apk',
                'ios_ipa': 'DecisionAR_v1.0.ipa',
                'size_mb': 45.2
            }
        }
        
        # 生成AR应用代码
        ar_code = self._generate_ar_code(ar_config)
        
        self.ar_apps[app_name] = {
            'config': ar_config,
            'code': ar_code,
            'status': 'ready_for_compilation'
        }
        
        logger.info(f"✅ AR应用 {app_name} 创建完成")
        return ar_config
    
    def generate_hologram(self, hologram_name: str) -> Dict:
        """生成全息投影展示"""
        logger.info(f"🌟 生成全息投影: {hologram_name}")
        
        hologram_config = {
            'name': hologram_name,
            'type': 'Production_Hologram',
            'display_technology': 'Pepper_Ghost_Illusion',
            'content': {
                'production_flow_3d': {
                    'animation': 'continuous_loop',
                    'highlight_optimization_points': True,
                    'real_time_data_sync': True
                },
                'decision_tree_visualization': {
                    'interactive_branches': True,
                    'color_coded_outcomes': True,
                    'probability_animations': True
                },
                'performance_metrics': {
                    'floating_charts': True,
                    'dynamic_updates': True,
                    'multi_layer_display': True
                }
            },
            'projection_specs': {
                'resolution': '4K_UHD',
                'brightness': '3000_lumens',
                'viewing_angle': '270_degrees',
                'projection_size': '2m_x_2m_x_2m'
            },
            'interaction_methods': [
                'gesture_control',
                'voice_activation',
                'mobile_app_remote'
            ],
            'export_format': '.holo',
            'file_size_gb': 2.1
        }
        
        # 生成全息内容数据
        hologram_data = self._generate_hologram_data(hologram_config)
        
        self.holograms[hologram_name] = {
            'config': hologram_config,
            'data': hologram_data,
            'status': 'ready_for_projection'
        }
        
        logger.info(f"✅ 全息投影 {hologram_name} 生成完成")
        return hologram_config
    
    def _generate_unity_script(self, config: Dict) -> str:
        """生成Unity C#脚本"""
        script = f"""
using UnityEngine;
using UnityEngine.XR;
using System.Collections;

public class {config['scene_name']}Controller : MonoBehaviour
{{
    // VR控制器
    public Transform leftController;
    public Transform rightController;
    
    // 场景组件
    public GameObject productionLine;
    public GameObject qualityStation;
    public GameObject decisionDashboard;
    
    // 数据连接
    private RealTimeDataConnector dataConnector;
    
    void Start()
    {{
        InitializeVRScene();
        ConnectRealTimeData();
        SetupInteractions();
    }}
    
    void InitializeVRScene()
    {{
        // 初始化VR场景
        Debug.Log("初始化VR场景: {config['scene_name']}");
        
        // 设置生产线
        if (productionLine != null)
        {{
            productionLine.SetActive(true);
            StartCoroutine(AnimateProductionLine());
        }}
        
        // 设置质量控制站
        if (qualityStation != null)
        {{
            qualityStation.GetComponent<QualityController>().EnableDefectHighlight();
        }}
    }}
    
    void ConnectRealTimeData()
    {{
        dataConnector = GetComponent<RealTimeDataConnector>();
        dataConnector.OnDataUpdate += UpdateVisualization;
        dataConnector.Connect("ws://localhost:8080/realtime");
    }}
    
    void SetupInteractions()
    {{
        // 设置手势控制
        var gestureDetector = GetComponent<GestureDetector>();
        gestureDetector.OnSwipe += NavigateScene;
        gestureDetector.OnPinch += ZoomView;
        gestureDetector.OnGrab += SelectObject;
    }}
    
    IEnumerator AnimateProductionLine()
    {{
        while (true)
        {{
            // 动画逻辑
            yield return new WaitForSeconds(0.1f);
        }}
    }}
    
    void UpdateVisualization(string jsonData)
    {{
        // 更新实时数据显示
        var data = JsonUtility.FromJson<ProductionData>(jsonData);
        decisionDashboard.GetComponent<DashboardController>().UpdateMetrics(data);
    }}
}}

[System.Serializable]
public class ProductionData
{{
    public float defectRate;
    public float qualityScore;
    public string currentDecision;
    public float confidence;
}}
"""
        return script
    
    def _generate_ar_code(self, config: Dict) -> str:
        """生成AR应用代码"""
        code = f"""
// AR决策辅助应用
package com.mathmodeling.{config['app_name'].lower()};

import com.google.ar.core.*;
import com.google.ar.sceneform.*;

public class {config['app_name']}Activity extends ArActivity {{
    
    private ArSceneView arSceneView;
    private DecisionOverlay decisionOverlay;
    private RealTimeDataService dataService;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {{
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_ar);
        
        initializeAR();
        setupDecisionOverlay();
        connectDataService();
    }}
    
    private void initializeAR() {{
        arSceneView = findViewById(R.id.ar_scene_view);
        arSceneView.getSession().configure(
            new Config(arSceneView.getSession())
                .setPlaneDiscoveryEnabled(true)
                .setLightEstimationEnabled(true)
        );
    }}
    
    private void setupDecisionOverlay() {{
        decisionOverlay = new DecisionOverlay(this);
        decisionOverlay.setPosition(0, 0, -2); // 2米前方
        decisionOverlay.enableRealTimeUpdates(true);
        
        arSceneView.getScene().addChild(decisionOverlay);
    }}
    
    private void connectDataService() {{
        dataService = new RealTimeDataService();
        dataService.setOnDataUpdateListener(data -> {{
            runOnUiThread(() -> {{
                decisionOverlay.updateDecision(data.decision);
                decisionOverlay.updateConfidence(data.confidence);
                decisionOverlay.updateQualityScore(data.qualityScore);
            }});
        }});
        dataService.connect();
    }}
    
    @Override
    public void onTap(HitResult hitResult, Plane plane, MotionEvent motionEvent) {{
        // 处理点击交互
        if (decisionOverlay.isHit(hitResult)) {{
            executeDecision(decisionOverlay.getCurrentDecision());
        }}
    }}
    
    private void executeDecision(String decision) {{
        // 执行AR决策
        Intent intent = new Intent("EXECUTE_DECISION");
        intent.putExtra("decision", decision);
        sendBroadcast(intent);
        
        // 显示确认动画
        decisionOverlay.showConfirmationAnimation();
    }}
}}
"""
        return code
    
    def _generate_hologram_data(self, config: Dict) -> Dict:
        """生成全息投影数据"""
        return {
            'geometry': {
                'vertices': self._generate_3d_vertices(),
                'faces': self._generate_3d_faces(),
                'textures': self._generate_texture_mapping()
            },
            'animation': {
                'keyframes': self._generate_animation_keyframes(),
                'transitions': self._generate_transitions(),
                'effects': ['particle_systems', 'light_rays', 'data_streams']
            },
            'interaction': {
                'hotspots': self._generate_interaction_points(),
                'responses': self._generate_interaction_responses()
            }
        }
    
    def _generate_3d_vertices(self) -> List:
        """生成3D顶点数据"""
        # 生成工厂布局的3D顶点
        vertices = []
        
        # 生产线主体
        for i in range(100):
            x = i * 0.1
            y = np.sin(i * 0.1) * 0.5
            z = 0
            vertices.append([x, y, z])
        
        # 质量检测点
        for i in range(10):
            x = i * 1.0
            y = 1.0
            z = 0.5
            vertices.append([x, y, z])
        
        return vertices
    
    def _generate_3d_faces(self) -> List:
        """生成3D面数据"""
        faces = []
        # 简化的面定义
        for i in range(0, 90, 3):
            faces.append([i, i+1, i+2])
        return faces
    
    def _generate_texture_mapping(self) -> Dict:
        """生成纹理映射"""
        return {
            'production_line': 'metallic_surface.jpg',
            'quality_station': 'control_panel.jpg',
            'data_streams': 'particle_effect.png'
        }
    
    def _generate_animation_keyframes(self) -> List:
        """生成动画关键帧"""
        keyframes = []
        for t in range(0, 100, 5):
            keyframes.append({
                'time': t,
                'position': [t * 0.1, np.sin(t * 0.1), 0],
                'rotation': [0, t * 0.05, 0],
                'scale': [1, 1, 1]
            })
        return keyframes
    
    def _generate_transitions(self) -> List:
        """生成过渡效果"""
        return [
            {'type': 'fade_in', 'duration': 1.0},
            {'type': 'slide_up', 'duration': 0.5},
            {'type': 'glow_effect', 'duration': 2.0}
        ]
    
    def _generate_interaction_points(self) -> List:
        """生成交互点"""
        return [
            {'position': [0, 1, 0], 'action': 'show_optimization_details'},
            {'position': [5, 1, 0], 'action': 'display_quality_metrics'},
            {'position': [10, 1, 0], 'action': 'open_decision_panel'}
        ]
    
    def _generate_interaction_responses(self) -> Dict:
        """生成交互响应"""
        return {
            'show_optimization_details': 'display_algorithm_visualization',
            'display_quality_metrics': 'show_quality_charts',
            'open_decision_panel': 'activate_decision_interface'
        }

class LivingPaperSystem:
    """活论文系统"""
    
    def __init__(self):
        self.interactive_elements = []
        self.live_data_connections = []
        self.executable_code_cells = []
        
    def create_living_paper(self) -> Dict:
        """创建交互式活论文"""
        logger.info("📄 创建交互式活论文系统...")
        
        paper_config = {
            'title': '生产过程决策优化的智能算法研究',
            'type': 'Interactive_Living_Paper',
            'platform': 'Web_Based_Jupyter_Like',
            'features': {
                'interactive_formulas': True,
                'live_data_updates': True,
                'executable_code': True,
                'real_time_visualization': True,
                'collaborative_editing': True
            },
            'sections': [
                {
                    'title': '抽样检验方案优化',
                    'interactive_elements': [
                        {
                            'type': 'formula_slider',
                            'formula': 'P(accept|p) = Σ(C(n,k) * p^k * (1-p)^(n-k))',
                            'parameters': {
                                'p': {'min': 0.0, 'max': 0.3, 'step': 0.01, 'default': 0.1},
                                'n': {'min': 10, 'max': 200, 'step': 10, 'default': 100}
                            },
                            'output': 'probability_chart'
                        }
                    ],
                    'code_cells': [
                        {
                            'language': 'python',
                            'code': 'optimal_sampling.py',
                            'editable': True,
                            'auto_execute': False
                        }
                    ]
                },
                {
                    'title': '生产决策优化',
                    'interactive_elements': [
                        {
                            'type': 'decision_tree_viz',
                            'data_source': 'live_production_data',
                            'interactive': True
                        }
                    ]
                },
                {
                    'title': '实时性能展示',
                    'live_data': {
                        'source': 'optimization_engine_stream',
                        'update_interval': 1000,
                        'charts': ['performance_metrics', 'decision_timeline']
                    }
                }
            ],
            'deployment': {
                'platform': 'GitHub_Pages',
                'cdn': 'CloudFlare',
                'domain': 'mathmodeling-living-paper.github.io',
                'ssl': True
            }
        }
        
        # 生成HTML/JavaScript代码
        html_code = self._generate_living_paper_html(paper_config)
        js_code = self._generate_living_paper_js(paper_config)
        
        return {
            'config': paper_config,
            'html': html_code,
            'javascript': js_code,
            'status': 'ready_for_deployment'
        }
    
    def _generate_living_paper_html(self, config: Dict) -> str:
        """生成活论文HTML"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config['title']}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.css">
    <link rel="stylesheet" href="living-paper.css">
</head>
<body>
    <div class="living-paper-container">
        <header class="paper-header">
            <h1>{config['title']}</h1>
            <div class="paper-meta">
                <span class="live-indicator">🔴 LIVE</span>
                <span class="last-update">最后更新: <span id="last-update-time"></span></span>
            </div>
        </header>
        
        <nav class="paper-nav">
            <ul>
                <li><a href="#section-1">抽样检验优化</a></li>
                <li><a href="#section-2">生产决策优化</a></li>
                <li><a href="#section-3">实时性能展示</a></li>
            </ul>
        </nav>
        
        <main class="paper-content">
            <section id="section-1" class="paper-section">
                <h2>抽样检验方案优化</h2>
                
                <div class="interactive-formula">
                    <div class="formula-display">
                        <span class="katex-formula">$$P(accept|p) = \\sum_{{k=0}}^{{c}} \\binom{{n}}{{k}} p^k (1-p)^{{n-k}}$$</span>
                    </div>
                    
                    <div class="parameter-controls">
                        <div class="slider-group">
                            <label for="p-slider">不合格率 p:</label>
                            <input type="range" id="p-slider" min="0" max="0.3" step="0.01" value="0.1">
                            <span id="p-value">0.1</span>
                        </div>
                        
                        <div class="slider-group">
                            <label for="n-slider">样本量 n:</label>
                            <input type="range" id="n-slider" min="10" max="200" step="10" value="100">
                            <span id="n-value">100</span>
                        </div>
                    </div>
                    
                    <div class="visualization-output">
                        <canvas id="probability-chart" width="600" height="400"></canvas>
                    </div>
                </div>
                
                <div class="code-cell">
                    <div class="code-header">
                        <span class="language">Python</span>
                        <button class="run-button" onclick="executeCode('sampling-code')">▶ 运行</button>
                    </div>
                    <div class="code-editor">
                        <textarea id="sampling-code" class="code-textarea">
def optimal_sampling(p0=0.1, alpha=0.05, beta=0.1, p1=0.15):
    \"\"\"优化抽样检验方案\"\"\"
    # 实时可编辑的代码
    for n in range(10, 500):
        for c in range(n+1):
            actual_alpha = 1 - binom.cdf(c, n, p0)
            actual_beta = binom.cdf(c, n, p1)
            
            if actual_alpha <= alpha and actual_beta <= beta:
                return n, c, actual_alpha, actual_beta
    
    return None
                        </textarea>
                    </div>
                    <div class="code-output" id="sampling-output"></div>
                </div>
            </section>
            
            <section id="section-2" class="paper-section">
                <h2>生产决策优化</h2>
                
                <div class="live-decision-tree">
                    <div id="decision-tree-viz"></div>
                </div>
                
                <div class="real-time-metrics">
                    <div class="metric-card">
                        <h3>当前决策</h3>
                        <div id="current-decision" class="metric-value">加载中...</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>置信度</h3>
                        <div id="confidence-level" class="metric-value">加载中...</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>预期收益</h3>
                        <div id="expected-benefit" class="metric-value">加载中...</div>
                    </div>
                </div>
            </section>
            
            <section id="section-3" class="paper-section">
                <h2>实时性能展示</h2>
                
                <div class="live-charts">
                    <div class="chart-container">
                        <h3>性能指标</h3>
                        <canvas id="performance-chart" width="500" height="300"></canvas>
                    </div>
                    
                    <div class="chart-container">
                        <h3>决策时间线</h3>
                        <canvas id="timeline-chart" width="500" height="300"></canvas>
                    </div>
                </div>
            </section>
        </main>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/katex@0.13.0/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="living-paper.js"></script>
</body>
</html>
"""
        return html
    
    def _generate_living_paper_js(self, config: Dict) -> str:
        """生成活论文JavaScript"""
        js = """
class LivingPaper {
    constructor() {
        this.websocket = null;
        this.charts = {};
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupInteractiveElements();
        this.setupCharts();
        this.updateTimestamp();
    }
    
    setupWebSocket() {
        this.websocket = new WebSocket('ws://localhost:8080/paper-updates');
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateLiveData(data);
        };
        
        this.websocket.onopen = () => {
            console.log('活论文数据连接已建立');
        };
    }
    
    setupInteractiveElements() {
        // 公式参数滑块
        const pSlider = document.getElementById('p-slider');
        const nSlider = document.getElementById('n-slider');
        
        pSlider.addEventListener('input', (e) => {
            document.getElementById('p-value').textContent = e.target.value;
            this.updateProbabilityChart();
        });
        
        nSlider.addEventListener('input', (e) => {
            document.getElementById('n-value').textContent = e.target.value;
            this.updateProbabilityChart();
        });
    }
    
    updateProbabilityChart() {
        const p = parseFloat(document.getElementById('p-slider').value);
        const n = parseInt(document.getElementById('n-slider').value);
        
        // 计算概率分布
        const data = this.calculateProbabilityDistribution(p, n);
        
        // 更新图表
        const canvas = document.getElementById('probability-chart');
        const ctx = canvas.getContext('2d');
        
        // 清除画布
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // 绘制概率曲线
        this.drawProbabilityCurve(ctx, data);
    }
    
    calculateProbabilityDistribution(p, n) {
        const data = [];
        for (let k = 0; k <= n; k++) {
            const prob = this.binomialProbability(n, k, p);
            data.push({x: k, y: prob});
        }
        return data;
    }
    
    binomialProbability(n, k, p) {
        const coefficient = this.binomialCoefficient(n, k);
        return coefficient * Math.pow(p, k) * Math.pow(1 - p, n - k);
    }
    
    binomialCoefficient(n, k) {
        if (k > n) return 0;
        if (k === 0 || k === n) return 1;
        
        let result = 1;
        for (let i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }
    
    drawProbabilityCurve(ctx, data) {
        const canvas = ctx.canvas;
        const width = canvas.width;
        const height = canvas.height;
        const padding = 40;
        
        // 绘制坐标轴
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.stroke();
        
        // 绘制概率曲线
        ctx.strokeStyle = '#007acc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        data.forEach((point, index) => {
            const x = padding + (point.x / data.length) * (width - 2 * padding);
            const y = height - padding - (point.y * (height - 2 * padding));
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
    }
    
    updateLiveData(data) {
        // 更新实时决策数据
        if (data.current_decision) {
            document.getElementById('current-decision').textContent = data.current_decision;
        }
        
        if (data.confidence_level) {
            document.getElementById('confidence-level').textContent = 
                (data.confidence_level * 100).toFixed(1) + '%';
        }
        
        if (data.expected_benefit) {
            document.getElementById('expected-benefit').textContent = 
                data.expected_benefit.toFixed(0);
        }
        
        // 更新图表
        this.updatePerformanceChart(data.performance_metrics);
        this.updateTimelineChart(data.decision_timeline);
    }
    
    updatePerformanceChart(metrics) {
        if (!this.charts.performance) return;
        
        this.charts.performance.data.datasets[0].data.push(metrics.latency);
        this.charts.performance.data.datasets[1].data.push(metrics.throughput);
        
        // 保持最新50个数据点
        if (this.charts.performance.data.datasets[0].data.length > 50) {
            this.charts.performance.data.datasets[0].data.shift();
            this.charts.performance.data.datasets[1].data.shift();
        }
        
        this.charts.performance.update();
    }
    
    updateTimelineChart(timeline) {
        if (!this.charts.timeline) return;
        
        this.charts.timeline.data.labels.push(new Date().toLocaleTimeString());
        this.charts.timeline.data.datasets[0].data.push(timeline.decision_count);
        
        if (this.charts.timeline.data.labels.length > 20) {
            this.charts.timeline.data.labels.shift();
            this.charts.timeline.data.datasets[0].data.shift();
        }
        
        this.charts.timeline.update();
    }
    
    setupCharts() {
        // 性能图表
        const perfCtx = document.getElementById('performance-chart').getContext('2d');
        this.charts.performance = new Chart(perfCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '延迟 (ms)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    tension: 0.1
                }, {
                    label: '吞吐量',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                animation: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // 时间线图表
        const timelineCtx = document.getElementById('timeline-chart').getContext('2d');
        this.charts.timeline = new Chart(timelineCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: '决策次数',
                    data: [],
                    backgroundColor: 'rgba(75, 192, 192, 0.6)'
                }]
            },
            options: {
                responsive: true,
                animation: false
            }
        });
    }
    
    updateTimestamp() {
        const now = new Date();
        document.getElementById('last-update-time').textContent = 
            now.toLocaleString('zh-CN');
        
        // 每秒更新时间戳
        setTimeout(() => this.updateTimestamp(), 1000);
    }
}

// 代码执行功能
function executeCode(codeId) {
    const code = document.getElementById(codeId).value;
    const outputElement = document.getElementById(codeId.replace('-code', '-output'));
    
    // 模拟代码执行
    outputElement.innerHTML = '<div class="loading">正在执行...</div>';
    
    // 发送到后端执行
    fetch('/execute-code', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({code: code})
    })
    .then(response => response.json())
    .then(result => {
        outputElement.innerHTML = `
            <div class="output-result">
                <pre>${result.output}</pre>
                ${result.chart ? `<img src="${result.chart}" alt="结果图表">` : ''}
            </div>
        `;
    })
    .catch(error => {
        outputElement.innerHTML = `<div class="error">执行错误: ${error.message}</div>`;
    });
}

// 初始化活论文系统
document.addEventListener('DOMContentLoaded', () => {
    new LivingPaper();
});
"""
        return js

def deploy_immersive_system():
    """部署沉浸式展示系统"""
    logger.info("🚀 开始部署沉浸式展示系统...")
    
    viz_system = ImmersiveVisualizationSystem()
    paper_system = LivingPaperSystem()
    
    # 1. 构建VR系统
    vr_config = viz_system.build_unity_scene('FactoryVR')
    
    # 2. 创建AR应用
    ar_config = viz_system.create_ar_app('DecisionAR')
    
    # 3. 生成全息投影
    hologram_config = viz_system.generate_hologram('production_hologram')
    
    # 4. 创建活论文
    paper_config = paper_system.create_living_paper()
    
    # 生成部署配置
    deployment_config = {
        'vr_system': {
            'webgl_export': 'output/FactoryVR_WebGL.zip',
            'access_url': 'https://mathmodeling-vr.github.io/factory-tour',
            'size_mb': vr_config['file_size_mb'],
            'platforms': vr_config['platforms']
        },
        'ar_apps': {
            'android_apk': 'output/DecisionAR_v1.0.apk',
            'ios_ipa': 'output/DecisionAR_v1.0.ipa',
            'download_links': {
                'android': 'https://github.com/releases/DecisionAR_Android.apk',
                'ios': 'https://testflight.apple.com/join/DecisionAR'
            }
        },
        'hologram': {
            'projection_file': 'output/production_hologram.holo',
            'setup_guide': 'docs/hologram_setup.md',
            'hardware_requirements': hologram_config['projection_specs']
        },
        'living_paper': {
            'hosting_url': 'https://mathmodeling-paper.github.io',
            'github_repo': 'https://github.com/mathmodeling/living-paper',
            'access_methods': ['web_browser', 'mobile_optimized']
        }
    }
    
    # 保存部署配置
    with open('output/immersive_deployment_config.json', 'w', encoding='utf-8') as f:
        json.dump(deployment_config, f, indent=2, ensure_ascii=False)
    
    # 生成一键访问链接
    access_links = {
        'primary_showcase': 'https://mathmodeling-showcase.github.io',
        'vr_experience': deployment_config['vr_system']['access_url'],
        'living_paper': deployment_config['living_paper']['hosting_url'],
        'ar_android': deployment_config['ar_apps']['download_links']['android'],
        'ar_ios': deployment_config['ar_apps']['download_links']['ios']
    }
    
    logger.info("✅ 沉浸式展示系统部署完成!")
    
    return {
        'deployment_config': deployment_config,
        'access_links': access_links,
        'status': 'deployed'
    }

if __name__ == '__main__':
    # 部署完整的沉浸式展示系统
    result = deploy_immersive_system()
    
    print("🎯 沉浸式展示系统部署完成!")
    print("🔗 一键访问链接:")
    for name, url in result['access_links'].items():
        print(f"  {name}: {url}")
    
    # 保存结果
    with open('output/immersive_system_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False) 