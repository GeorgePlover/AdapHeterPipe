# 生成可视化结果

from typing import List, Dict
from simulator import Simulator, SimConf, Task

# Task 可视化相关信息：microbatch_id, stage_id, type(F,B,W), start_time, end_time, worker_id
# 绘制甘特图，横轴为时间，纵轴为worker_id，不同颜色表示不同类型的任务，F用蓝色，B用橙色，W用绿色
# stage_id 利用颜色的渐变来表示虚拟流水线的深度，即stage_id越大颜色越深

def generate_gantt_chart(tasklist: List['Task'], filename: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(12, 4))

    # 基础颜色映射
    base_color_map = {
        "F": "blue",
        "B": "orange", 
        "W": "green"
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
            
        # 根据stage_id计算颜色深浅 (0到1之间的值)
        color_intensity = 1- task.stage_id / max_stage if max_stage > 0 else 0.5
        
        # 获取对应类型的颜色
        task_color = color_maps[task.task_type](color_intensity)
        
        # 绘制矩形条
        ax.barh(
            y=task.worker_id,
            width=duration,
            left=task.start_time,
            height=0.4,
            color=task_color,
            edgecolor='black',
            linewidth=0.5,
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
                fontsize=8,
                color='white' if color_intensity > 0.6 else 'black'
            )
    
    # 设置图表属性
    ax.set_xlabel('Time')
    ax.set_ylabel('Worker ID')
    ax.set_title('Gantt Chart of Tasks')
    
    # 设置y轴为整数
    if tasklist:
        worker_ids = sorted(set(task.worker_id for task in tasklist),reverse=True)
        ax.set_yticks(worker_ids)
        
    # y坐标轴翻转
    ax.invert_yaxis()
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3)
    
    # 创建图例
    f_patch = mpatches.Patch(color='blue', label='Forward (F)')
    b_patch = mpatches.Patch(color='orange', label='Backward (B)') 
    w_patch = mpatches.Patch(color='green', label='Weight Update (W)')
    ax.legend(handles=[f_patch, b_patch, w_patch], loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.15))
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()