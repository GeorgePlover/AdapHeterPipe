============================================================
  AutoHeterPipe 甘特图独立绘制工具
============================================================

本文件夹包含 5 种流水线调度策略的甘特图独立绘制代码。

调度策略:
  - GPipe        : 同步流水线
  - 1F1B         : 壹前壹后流水线
  - Interleaved 1F1B: 交错式 1F1B
  - ZB           : 零气泡流水线
  - ZB-V         : V形零气泡流水线

============================================================
  使用方法
============================================================

1. 安装依赖:
   pip install matplotlib

2. 运行:
   python plot_gantt.py

3. 输出:
   output/ 目录下生成 5 张 PNG 甘特图 (600 DPI)

============================================================
  自定义修改说明
============================================================

打开 plot_gantt.py，顶部 "可调参数区" 提供以下参数:

[颜色]
  COLOR_FORWARD   = "#5CA9C8"  # 前向传播颜色
  COLOR_BACKWARD  = "#5DB558"  # 反向传播颜色
  COLOR_WEIGHT    = "#C77337"  # 权重梯度计算颜色
  BAR_EDGE_COLOR  = "black"    # 条形边框颜色
  BAR_ALPHA       = 0.8        # 条形透明度
  BAR_HATCH       = ""         # 条形纹理 (可选: / \ x)

[字号]
  FONT_SIZE_BAR_LABEL = 14    # 条形内编号
  FONT_SIZE_XLABEL    = 18    # X轴标签
  FONT_SIZE_YTICK     = 18    # Y轴刻度
  FONT_SIZE_LEGEND    = 18    # 图例

[尺寸]
  FIGURE_WIDTH  = 12          # 图片宽度 (英寸)
  FIGURE_HEIGHT = 4           # 图片高度 (英寸)
  DPI           = 600         # 分辨率

[其他]
  SHOW_MICROBATCH_IDS = True  # 是否显示 microbatch 编号

[字体]
  默认: 中文 宋体(SimSun), 英文 Times New Roman
  脚本会自动检测系统可用字体并降级，启动时会打印实际使用的字体。
  如需强制指定字体，修改 CN_FONT_CANDIDATES / EN_FONT_CANDIDATES 列表。
  粗体策略:
    - 系统有原生粗体字重 (weight>=600): 自动使用原生粗体
    - 系统无粗体字重: 使用 path_effects 描边模拟粗体
  调整描边宽度: BOLD_STROKE_WIDTH (默认 1.2, 设为 0 禁用)
  提示: Windows 上 SimSun 通常自带 Bold, 无需描边。

============================================================
  文件结构
============================================================

draw-pack/
├── data/                     # 任务调度数据 (JSON)
│   ├── GPipe_draw_task_data.json
│   ├── 1F1B_draw_task_data.json
│   ├── Interleaved_1F1B_draw_task_data.json
│   ├── ZB_draw_task_data.json
│   └── ZB_Vshape_draw_task_data.json
├── plot_gantt.py             # 主脚本
├── output/                   # (运行后生成) 输出 PNG
│   ├── GPipe_gantt.png
│   ├── 1F1B_gantt.png
│   ├── Interleaved_1F1B_gantt.png
│   ├── ZB_gantt.png
│   └── ZB_Vshape_gantt.png
└── README.txt                # 本文件
