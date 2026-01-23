# 生成可视化结果

from typing import List, Dict
from simulator import Simulator, SimConf, Task


def plot_pipeline_metrics(data, output_dir='./plots'):
    """简化的流水线指标可视化"""

    import matplotlib.pyplot as plt
    from pathlib import Path
    # 加载数据

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for method in list(data):
        if "record" not in method:
            data.remove(method)
    
    for idx, method in enumerate(data):
        method_name = method['name']
        records = method['record']
        
        stages = range(len(records))
        overlap = [r['overlapping_time_ratio'] for r in records]
        bubble = [100.0 + r['overlapping_time_ratio'] - r['sending_time_ratio'] - r['computing_time_ratio'] for r in records]
        memory = [r['peak_mem_rate'] * 100 for r in records]
        
        # 通信掩盖率
        axes[0].plot(stages, overlap, 'o-', label=method_name, 
                    color=colors[idx % len(colors)], linewidth=2, markersize=8)
        
        # 气泡率
        axes[1].bar([s + idx*0.15 for s in stages], bubble, width=0.15,
                   label=method_name, color=colors[idx % len(colors)])
        
        # 峰值内存率
        axes[2].plot(stages, memory, 's--', label=method_name,
                    color=colors[idx % len(colors)], linewidth=2, markersize=8)
    
    # 设置图表属性
    titles = ['Communication Overlap Ratio', 'Bubble Rate', 'Peak Memory Ratio']
    ylabels = ['Ratio (%)', 'Rate (%)', 'Memory Ratio (%)']
    
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel('Pipeline Stage', fontsize=12)
        ax.set_ylabel(ylabels[i], fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='best')
    
    # 设置x轴标签
    stage_count = max(len(method['record']) for method in data)
    for ax in axes:
        ax.set_xticks(range(stage_count))
        ax.set_xticklabels([f'Stage {i+1}' for i in range(stage_count)])
    
    plt.tight_layout()
    
    # 保存图表
    output_file = f'{output_dir}/pipeline_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.show()


# Task 可视化相关信息：microbatch_id, stage_id, type(F,B,W), start_time, end_time, worker_id
# 绘制甘特图，横轴为时间，纵轴为worker_id，不同颜色表示不同类型的任务，F用蓝色，B用橙色，W用绿色
# stage_id 利用颜色的渐变来表示虚拟流水线的深度，即stage_id越大颜色越深

def generate_gantt_chart(tasklist: List['Task'], filename: str,
                         no_W:bool = False):
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
    ncol = 2
    f_patch = mpatches.Patch(color='blue', label='Forward (F)')
    b_patch = mpatches.Patch(color='orange', label='Backward (B)') 
    handles = [f_patch, b_patch]
    if no_W is False:
        w_patch = mpatches.Patch(color='green', label='Weight Update (W)')
        ncol=3
        handles.append(w_patch)
    
    ax.legend(handles=handles, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, -0.15))
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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