# Golf Swing Analyzer

基于 AI 的高尔夫挥杆分析工具，使用 MediaPipe 进行姿态估计，自动检测挥杆阶段并给出改进建议。

## 功能特点

- **自动姿态检测** - 使用 MediaPipe PoseLandmarker 提取 33 个身体关键点
- **挥杆阶段识别** - 自动识别 Setup、Backswing、Top、Downswing、Impact、Follow-through
- **关键指标计算** - 脊柱角度、X-Factor、左臂角度、头部稳定性等
- **可视化输出** - 骨骼叠加视频、关键帧截图、指标曲线图
- **智能建议** - 基于分析结果生成改进建议

## 安装

### 环境要求

- Python 3.10+
- macOS (M1/M2) 或 Linux
- ffmpeg

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/xurui1982/golf_demo.git
cd golf_demo

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
# 激活虚拟环境
source venv/bin/activate

# 分析视频
python -m src.main your_video.mp4 -o output/
```

### 命令行参数

```
python -m src.main <video_path> [options]

参数:
  video              输入视频文件路径
  -o, --output       输出目录 (默认: output)
  --no-ball          不追踪球
  --no-video         不生成标注视频
  -q, --quiet        安静模式
```

### 示例

```bash
# 完整分析
python -m src.main swing.mp4 -o results/

# 只分析姿态，不追踪球
python -m src.main swing.mp4 -o results/ --no-ball

# 快速分析，不生成视频
python -m src.main swing.mp4 -o results/ --no-video -q
```

## 输出文件

分析完成后，会在输出目录生成以下文件：

```
output/
├── keyframes/            # 关键帧截图（带骨骼标注）
│   ├── setup.jpg
│   ├── backswing.jpg
│   ├── top.jpg
│   ├── downswing.jpg
│   ├── impact.jpg
│   └── follow_through.jpg
├── charts/
│   └── swing_metrics.png # 挥杆指标曲线图
├── data/
│   ├── pose_data.json    # 姿态关键点数据
│   ├── ball_track.json   # 球追踪数据（如有）
│   └── metrics.json      # 计算指标
├── annotated.mp4         # 带骨骼叠加的视频
└── recommendation.md     # 分析建议报告
```

## 视频拍摄建议

### 推荐设置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 帧率 | 240fps | iPhone 慢动作模式，用于球追踪 |
| 分辨率 | 1080p+ | 确保细节清晰 |
| 时长 | 3-10秒 | 单次完整挥杆 |

### 拍摄角度

- **侧面视角 (Face-on)** - 面对球手拍摄，适合分析重心转移、髋部旋转
- **背面视角 (Down-the-line)** - 从球飞行方向后方拍摄，适合分析挥杆平面

### 拍摄技巧

- 全身入镜，头顶和脚下留有空间
- 球和球手都在画面中
- 背景尽量简洁
- 光线充足，避免逆光

## 分析指标说明

| 指标 | 说明 | 参考范围 |
|------|------|----------|
| Spine Angle | 脊柱前倾角度 | 40-50° |
| X-Factor | 髋肩分离角度 | 35-45° |
| Left Arm Angle (Top) | 顶点时左臂角度 | >160° (越接近180°越好) |
| Head Movement | 头部移动量 | <0.3 (越小越稳定) |
| Tempo Ratio | 上杆/下杆时间比 | 约 3:1 |

## 项目结构

```
golf_demo/
├── src/
│   ├── main.py              # 主入口
│   ├── video_processor.py   # 视频处理
│   ├── pose_analyzer.py     # 姿态分析
│   ├── ball_tracker.py      # 球追踪
│   ├── swing_detector.py    # 挥杆阶段检测
│   ├── metrics.py           # 指标计算
│   └── visualizer.py        # 可视化
├── models/
│   └── pose_landmarker_heavy.task  # MediaPipe 模型
├── docs/plans/
│   └── 2026-01-29-golf-swing-analyzer-design.md
├── output/                  # 分析输出
├── requirements.txt
└── README.md
```

## 技术栈

- **MediaPipe** - 姿态估计
- **OpenCV** - 图像处理、球追踪、可视化
- **NumPy** - 数值计算
- **Matplotlib** - 图表生成

## License

MIT
