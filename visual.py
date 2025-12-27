# 生成可视化结果

from typing import List, Dict
from simulator import Simulator, SimConf, Task

# Task 可视化相关信息：microbatch_id, stage_id, type(F,B,W), start_time, end_time, worker_id
# 绘制甘特图，横轴为时间，纵轴为worker_id，不同颜色表示不同类型的任务，F用蓝色，B用橙色，W用绿色
# stage_id 利用颜色的渐变来表示虚拟流水线的深度，即stage_id越大颜色越深

def generate_gantt_chart(tasklist: List['Task'], filename: str,
                         no_W:bool = False, pipeline_schedule_type: str = "1F1B"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    stage_cnt = max(task.stage_id for task in tasklist) + 1 if tasklist else 1
    fig, ax = plt.subplots(figsize=(12, 4))

    # 基础颜色映射
    base_color_map = {
        "F": "#BFBFBF",#"#AAAAAA",
        "B": "#3F3F3F",#"#555555", 
        "W": "#3F3F3F"#"#555555"
    }
    
    # 找出最大的stage_id用于颜色渐变
    max_stage = max(task.stage_id for task in tasklist) if tasklist else 1
    
    # 为每个任务类型创建颜色映射
    color_maps = {}
    for task_type, base_color in base_color_map.items():
        # 创建从浅到深的颜色映射
        base_rgb = mcolors.to_rgb(base_color)
        light_color = (base_rgb[0]*0.4 + 0.6, base_rgb[1]*0.4 + 0.6, base_rgb[2]*0.4 + 0.6)  # 浅色
        dark_color = base_rgb  # 深色
        color_maps[task_type] = mcolors.LinearSegmentedColormap.from_list(
            f"{task_type}_gradient", [light_color, dark_color]
        )
    
    # 绘制每个任务
    for task in tasklist:
        duration = task.end_time - task.start_time
        if duration <= 0:
            continue
        
        color_intensity = 1.0
        font_color = 'black'
        hatch = ''
        # 根据stage_id计算颜色深浅 (0到1之间的值)
        if pipeline_schedule_type == "1F1B":
            color_intensity = 1.0
        elif pipeline_schedule_type == "Gpipe":
            color_intensity = 1.0
        elif pipeline_schedule_type in ["Interleaved_1F1B", "ZB_V"]:
            if task.stage_id >= stage_cnt//2:
                font_color = 'white'
            else:
                color_intensity = 1.0
        else:
            color_intensity = 1.0
        
        if task.task_type == "W":
            hatch = 'xx'
        
        # 获取对应类型的颜色
        task_color = color_maps[task.task_type](color_intensity)
        
        # 绘制矩形条
        ax.barh(
            y=task.worker_id,
            width=duration,
            left=task.start_time,
            height=1.0,
            color=task_color,
            edgecolor='black',
            linewidth=1.5,
            hatch=hatch,
            alpha=0.8
        )
        
        # 添加任务类型标签（可选）
        if True or duration > 0.1 * (max(task.end_time for task in tasklist) if tasklist else 1):
            ax.text(
                task.start_time + duration/2,
                task.worker_id,
                task.microbatch_id.__str__(),
                ha='center',
                va='center',
                fontsize=14,
                weight='bold',
                color=font_color
            )
    
    # 设置图表属性，x轴文字靠右，不要居中
    ax.set_xlabel('Time', fontdict = {'size': 18, 'weight': 'bold'}, ha='right', x=0.95)
    # ax.set_ylabel('GPU-ID', fontdict = {'size': 18, 'weight': 'bold'}, ha="right", y=0.9)

    
    # ax.set_title('Gantt Chart of Tasks')
    
    # 设置y轴为GPU-整数
    if tasklist:
        worker_ids = sorted(set(task.worker_id for task in tasklist),reverse=True)
        worker_ids = [f"GPU-{wid}" for wid in worker_ids]
        # ax.set_yticks(worker_ids)
        ax.set_yticks(range(len(worker_ids)), labels=reversed(worker_ids))
        
    # y坐标轴翻转
    ax.invert_yaxis()
    
    # y轴坐标字体大小
    ax.tick_params(axis='y', labelsize=18)
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=18,
        weight='bold'
    )
    # 添加网格
    # ax.grid(True, axis='x', alpha=0.3)
    
    # 移除图表上左右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 下边框加粗，显示箭头
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['bottom'].set_capstyle('projecting')
    
    # 移除tick标记
    ax.tick_params(axis='both', which='both', length=0)
    
    # 移除x轴刻度数字
    
    ax.set_xticklabels([])
    
    
    # 创建图例
    ncol = 2
    f_patch = mpatches.Patch(facecolor=base_color_map["F"], label='Forward Pass', edgecolor='black')
    b_patch = mpatches.Patch(facecolor=base_color_map["B"], label='Backward Pass', edgecolor='black') 
    handles = [f_patch, b_patch]
    if no_W is False:
        w_patch = mpatches.Patch(facecolor=base_color_map["W"], label='Weight Pass', edgecolor='black', hatch='xx')
        ncol=3
        handles.append(w_patch)
    
    ax.legend(handles=handles, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, -0.05)
              , fontsize=18, frameon=False)
    # 加粗图例字体
    for text in ax.get_legend().get_texts():
        text.set_weight('bold')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Gantt chart saved to {filename}")
    

# 绘制实验结果柱状图
def generate_result_bar_chart(result_dict: Dict[str, float], filename: str):
    import matplotlib.pyplot as plt
    
    methods = list(result_dict.keys())
    throuputs = [result_dict[m] for m in methods]
    
    plt.bar(methods, throuputs)
    plt.xlabel('Method')
    plt.ylabel('Throughput (samples/s)')
    plt.title('Throughput Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Result bar chart saved to {filename}")