# 处理输入，输出profile数据

class Worker:
    def __init__(self, device_type):
        self.device_type = device_type
        pass
    
    def F_time_per_layer(self)-> float:
        # 返回该worker每层的前向时间
        return 1.0  # 示例值
    
    def B_time_per_layer(self) -> float:
        # 返回该worker每层的后向时间
        return 1.0  # 示例值
    
    def W_time_per_layer(self) -> float:
        # 返回该worker每层的权重更新时间
        return 1.0  # 示例值
    
    def time_per_layer(self, task_type: str) -> float:
        if task_type == "F":
            return self.F_time_per_layer()
        elif task_type == "B":
            return self.B_time_per_layer()
        elif task_type == "W":
            return self.W_time_per_layer()
        else:
            raise ValueError(f"Unknown task type: {task_type}")