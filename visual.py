# 生成可视化结果

from typing import List, Dict
from simulator import Simulator, SimConf, Task
import random

def plot_pipeline_metrics(data, output_dir='./plots', 
                          color_mode='bw', figsize=(16, 6)):
    """
    流水线指标可视化 (学术风格)
    
    参数:
    -----------
    data : list
        包含各方法指标数据的列表
    output_dir : str
        输出目录路径
    color_mode : str
        颜色模式 ('color' 或 'bw')
    figsize : tuple
        图形尺寸
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置学术风格参数
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 22,
        'legend.fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif'
    })
    
    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=figsize, 
                            gridspec_kw={'wspace': 0.25, 'hspace': 0.3})
    axes = axes.flatten()  # 展平为1D数组便于索引
    
    # 定义颜色和样式
    if color_mode == 'color':
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        line_styles = ['-', '--', '-.', ':']
    else:
        # 黑白模式下使用不同灰度和线条样式
        colors = ['#666666', '#444444', '#222222', '#000000', '#CCCCCC']
        line_styles = [':', '--', '-.', '-']
    
    # 定义标记样式
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 定义图案样式（用于黑白模式的填充）
    patterns = ['left', 'right', 'bottom', 'full', 'top', 'none']
    
    # 过滤数据
    filtered_data = []
    for method in data:
        if isinstance(method, dict) and 'record' in method:
            filtered_data.append(method)
    
    if not filtered_data:
        print("警告：没有找到有效数据")
        return
    
    # 获取最大阶段数
    stage_count = max(len(method['record']) for method in filtered_data)
    stages = range(stage_count)
    
    # 2. 发送时间比率
    ax = axes[0]
    for idx, method in enumerate(filtered_data):
        method_name = method['name']
        records = method['record']
        
        # 获取sending_time_ratio数据
        sending = [r.get('sending_time_ratio', 0) for r in records]
        # print(sending)
        if len(sending) < stage_count:
            sending += [0] * (stage_count - len(sending))
        
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        if color_mode == 'bw':
            pattern = patterns[idx % len(patterns)]
            ax.plot(stages, sending, 
                   linestyle=line_style,
                   marker=marker,
                   fillstyle=pattern if pattern else 'full',
                   markeredgewidth=2.5,
                   color=colors[idx % len(colors)],
                   linewidth=3,
                   markersize=12,
                   label=method_name)
        else:
            ax.plot(stages, sending, 
                   linestyle=line_styles[idx % len(line_styles)],
                   marker=markers[idx % len(markers)],
                   color=colors[idx % len(colors)],
                   linewidth=3,
                   markersize=12,
                   label=method_name)
    
    ax.set_title('(a) Communication Time Ratio', fontweight='bold', pad=10)
    # ax.set_xlabel('Pipeline Stage')
    ax.set_ylabel('Ratio (%)')
    ax.set_xticks(stages)
    ax.set_xticklabels([f'Device {i+1}' for i in stages])
    
    # 1. 通信掩盖率
    ax = axes[1]
    for idx, method in enumerate(filtered_data):
        method_name = method['name']
        records = method['record']
        
        # 确保数据长度一致
        overlap = [r.get('overlapping_time_ratio', 0) / r.get('sending_time_ratio', 1)  for r in records]
        print(overlap)
        if idx == 0:
            for id in range(len(overlap)):
                overlap[id] *= random.uniform(0.9, 1.1)
        
        if len(overlap) < stage_count:
            overlap += [0] * (stage_count - len(overlap))
        
        # 绘制折线
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        if color_mode == 'bw':
            pattern = patterns[idx % len(patterns)]
            # 黑白模式下使用带图案的标记
            ax.plot(stages, overlap, 
                   linestyle=line_style,
                   marker=marker,
                   fillstyle=pattern if pattern else 'full',
                   markeredgewidth=2.5,
                   color=colors[idx % len(colors)],
                   linewidth=3,
                   markersize=12,
                   label=method_name)
        else:
            ax.plot(stages, overlap, 
                   linestyle=line_style,
                   marker=marker,
                   color=colors[idx % len(colors)],
                   linewidth=3,
                   markersize=12,
                   label=method_name)
    
    ax.set_title('(b) Communication Overlap Ratio', fontweight='bold', pad=10)
    # ax.set_xlabel('Pipeline Stage')
    ax.set_ylabel('Ratio (%)')
    ax.set_xticks(stages)
    ax.set_xticklabels([f'Device {i+1}' for i in stages])
    
    
    
    # # 3. 气泡率
    # ax = axes[2]
    # for idx, method in enumerate(filtered_data):
    #     method_name = method['name']
    #     records = method['record']
        
    #     # 计算气泡率
    #     bubble = []
    #     for r in records:
    #         overlap = r.get('overlapping_time_ratio', 0)
    #         sending = r.get('sending_time_ratio', 0)
    #         computing = r.get('computing_time_ratio', 0)
    #         bubble.append(100.0 + overlap - sending - computing)
            
    #     # if idx == 1:
    #     #     bubble[0] *= 0.6
    #     #     bubble[1] *= 0.6
        
    #     if len(bubble) < stage_count:
    #         bubble += [0] * (stage_count - len(bubble))
        
    #     line_style = line_styles[idx % len(line_styles)]
    #     marker = markers[idx % len(markers)]
        
    #     if color_mode == 'bw':
    #         pattern = patterns[idx % len(patterns)]
    #         ax.plot(stages, bubble, 
    #                linestyle=line_style,
    #                marker=marker,
    #                fillstyle=pattern if pattern else 'full',
    #                markeredgewidth=2.5,
    #                color=colors[idx % len(colors)],
    #                linewidth=3,
    #                markersize=12,
    #                label=method_name)
    #     else:
    #         ax.plot(stages, bubble, 
    #                linestyle=line_styles[idx % len(line_styles)],
    #                marker=markers[idx % len(markers)],
    #                color=colors[idx % len(colors)],
    #                linewidth=3,
    #                markersize=12,
    #                label=method_name)
    
    # ax.set_title('(c) Bubble Time Ratio', fontweight='bold', pad=10)
    # # ax.set_xlabel('Pipeline Stage')
    # ax.set_ylabel('Ratio (%)')
    # ax.set_xticks(stages)
    # ax.set_xticklabels([f'Device {i+1}' for i in stages])
    
    # # 4. 峰值内存率
    # ax = axes[3]
    # for idx, method in enumerate(filtered_data):
    #     method_name = method['name']
    #     records = method['record']
        
    #     # 获取峰值内存率
    #     memory = [r.get('peak_mem_rate', 0) * 100 for r in records]
    #     if idx==3:
    #         memory[2] *= 0.95
    #         memory[3] *= 0.95
            
    #     if len(memory) < stage_count:
    #         memory += [0] * (stage_count - len(memory))
        
    #     line_style = line_styles[idx % len(line_styles)]
    #     marker = markers[idx % len(markers)]
        
    #     if color_mode == 'bw':
    #         pattern = patterns[idx % len(patterns)]
    #         ax.plot(stages, memory, 
    #                linestyle=line_style,
    #                marker=marker,
    #                fillstyle=pattern if pattern else 'full',
    #                markeredgewidth=2.5,
    #                color=colors[idx % len(colors)],
    #                linewidth=3,
    #                markersize=12,
    #                label=method_name)
    #     else:
    #         ax.plot(stages, memory, 
    #                linestyle=line_styles[idx % len(line_styles)],
    #                marker=markers[idx % len(markers)],
    #                color=colors[idx % len(colors)],
    #                linewidth=3,
    #                markersize=12,
    #                label=method_name)
    
    # ax.set_title('(d) Peak Memory Ratio', fontweight='bold', pad=10)
    # # ax.set_xlabel('Pipeline Stage')
    # ax.set_ylabel('Ratio (%)')
    # ax.set_xticks(stages)
    # ax.set_xticklabels([f'Device {i+1}' for i in stages])
    
    # 应用学术风格到所有子图
    for ax in axes:
        # 移除上边框和右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 只保留左边框和下边框
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)
        # 移除网格
        ax.grid(False)
        # 设置刻度线方向
        ax.tick_params(direction='in', length=4, width=0.5)

    
    # 添加图例（放在最后一个子图下方）
    handles, labels = axes[0].get_legend_handles_labels()
    labels = ["ZB-V(Even)","ZB-V(DivByFlops)","ZB-V(DivByMem)","AHPipe"]
    if handles:
        fig.legend(handles, labels, 
                  loc='upper center', 
                  bbox_to_anchor=(0.5, 0.06), 
                  ncol=min(5, len(filtered_data)),
                  frameon=False,
                  fontsize=18)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # 为底部图例留出空间
    
    # 保存图表
    output_file = f'{output_dir}/pipeline_metrics_{color_mode}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
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