# 处理输入，输出profile数据
from my_common import DEBUG
import json
TFLOP = 1e12

profiling_data = json.load(open("/home/nsh/nas/projects/Megatron-LM/device_profilings.json", "r"))

class Device:
    def __init__(self, name:str, tflops: float, memory_GB: float, tp:int = 1):
        self.name = name
        self.tflops = tflops * tp  # 设备的计算能力，单位为 TFLOP/s
        self.memory_GB = memory_GB * tp  # 设备的显存大小，单位为 GB
        self.tp = tp  # 设备的张量并行度
    
    def memory_bytes(self) -> float:
        return self.memory_GB * (1024.0 **3)  # 转换为 Bytes
    
    
class Model:
    def __init__(self,
                 name: str,
                 layer_num: int,
                 batch_size: int,
                 microbatch_size: int,
                 sequence_length: int,
                 hidden_size: int,
                 num_attention_heads: int,):
        self.name = name
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        # 显存单位 Bytes
        self.active_mem_per_layer_for_B = self.sequence_length * self.microbatch_size * \
            (34.0 * self.hidden_size + 5.0 * self.sequence_length * self.num_attention_heads)
        self.active_mem_per_layer_for_W = 32.0 * self.sequence_length * self.microbatch_size * self.hidden_size
        
    def static_mem_per_layer(self) -> float:
        return (12.0 * self.hidden_size + 13.0)* self.hidden_size * 16  # Bytes
    
    def active_mem_per_layer(self, task_type: str) -> float:
        # 每种计算任务类型对激活内存的消耗。负数表示释放，正数表示占用
        # 只关注一个microbatch的内存消耗
        if task_type == "F":
            return self.active_mem_per_layer_for_B
        elif task_type == "B":
            return self.active_mem_per_layer_for_W - self.active_mem_per_layer_for_B
        elif task_type == "W":
            return -self.active_mem_per_layer_for_W
    
    def flop16_per_layer(self, task_type: str) -> float:
        # 每种计算任务类型的每层计算量，单位为FLOP16
        # 示例值
        if task_type == "F":
            return self.sequence_length * self.microbatch_size * self.hidden_size * (24.0 * self.hidden_size + 4.0 * self.sequence_length)
        elif task_type == "B":
            return self.sequence_length * self.microbatch_size * self.hidden_size * (24.0 * self.hidden_size + 8.0 * self.sequence_length)
        elif task_type == "W":
            return self.sequence_length * self.microbatch_size * self.hidden_size * (24.0 * self.hidden_size)

    def print_info(self):
        print(f"Model: {self.name}, Layers: {self.layer_num}, Batch size: {self.batch_size}, "
              f"Microbatch size: {self.microbatch_size}, Seq length: {self.sequence_length}, "
              f"Hidden size: {self.hidden_size}, Attention heads: {self.num_attention_heads}")
        print(f"Active Mem per Layer for B: {self.active_mem_per_layer_for_B/(1024**3):.2f} GB")

class Worker:
    def __init__(self, device:Device, model:Model):
        self.device = device
        self.model = model
        
    def exist_profiling(self) ->bool:
        # 判断该worker是否存在profiling数据
        model_tp = self.model.name+"_tp"+str(self.device.tp)
        return (
            (self.device.name in profiling_data)
            and (model_tp in profiling_data[self.device.name])
            and self.model.microbatch_size == 1  # 目前只支持单个microbatch的情况
        )
    
    def get_profiling(self, task_type: str) -> float:
        # forward_time_per_layer , backward_time_per_layer
        if not self.exist_profiling():
            raise ValueError(f"No profiling data for {self.device.name} and {self.model.name}")
        model_tp = self.model.name+"_tp"+str(self.device.tp)
        return profiling_data[self.device.name][model_tp][task_type]
    
    def memory_limit(self) -> float: # Bytes
        # 返回该worker的内存限制
        return self.device.memory_GB * (1024.0 **3)  # 转换为Bytes
    
    def F_time_per_layer(self)-> float: # seconds
        # 返回该worker每层的前向时间
        if not self.exist_profiling():
            return self.model.flop16_per_layer("F") / (self.device.tflops * TFLOP)
        else:
            return self.get_profiling("forward_time_per_layer")
    
    def B_time_per_layer(self) -> float:
        # 返回该worker每层的后向时间
        if not self.exist_profiling():
            return self.model.flop16_per_layer("B") / (self.device.tflops * TFLOP)
        else:
            return self.get_profiling("backward_time_per_layer") * (self.model.flop16_per_layer("B") / (self.model.flop16_per_layer("B") + self.model.flop16_per_layer("W")))
    
    def W_time_per_layer(self) -> float:
        # 返回该worker每层的权重更新时间
        if not self.exist_profiling():
            return self.model.flop16_per_layer("W") / (self.device.tflops * TFLOP)
        else:
            return self.get_profiling("backward_time_per_layer") * (self.model.flop16_per_layer("W") / (self.model.flop16_per_layer("B") + self.model.flop16_per_layer("W")))
    
    def time_per_layer(self, task_type: str) -> float:
        if task_type == "F":
            return self.F_time_per_layer()
        elif task_type == "B":
            return self.B_time_per_layer()
        elif task_type == "W":
            return self.W_time_per_layer()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def active_mem_per_layer(self, task_type: str) -> float: # Bytes
        return self.model.active_mem_per_layer(task_type)
    
    def static_mem_per_layer(self) -> float: # Bytes
        return self.model.static_mem_per_layer()
    
def get_model(model_name: str) -> Model:
    # 根据模型名称返回对应的Model对象
    if model_name == "gpt2-small":
        res =  Model(
            name="gpt2-small",
            layer_num=16,
            batch_size=16,
            microbatch_size=2,
            sequence_length=1024,
            hidden_size=768,
            num_attention_heads=12
        )
    elif model_name == "gpt3_117m":
        res = Model(
            name="gpt3_117m",   # 类似 GPT-3 Ada 级别
            layer_num=12,
            batch_size=8,
            microbatch_size=1,
            sequence_length=1024,
            hidden_size=768,
            num_attention_heads=12
        )
    elif model_name == "gpt3_345m":
        res = Model(
            name="gpt3_345m",   # GPT-3 小模型
            layer_num=24,
            batch_size=8,
            microbatch_size=1,
            sequence_length=1024,
            hidden_size=1024,
            num_attention_heads=16
        )
    elif model_name == "gpt3_760m":
        res = Model(
            name="gpt3_760m",   # GPT-3 小模型
            layer_num=24,
            batch_size=8,
            microbatch_size=1,
            sequence_length=1024,
            hidden_size=1536,
            num_attention_heads=16
        )
    elif model_name == "gpt3_1.3b":
        res = Model(
            name="gpt3_1.3b",   # GPT-3 中型
            layer_num=24,
            batch_size=8,
            microbatch_size=1,
            sequence_length=2048,
            hidden_size=2048,
            num_attention_heads=16
        )
    elif model_name == "gpt3_6.7b":
        res = Model(
            name="gpt3_6.7b",   # GPT-3 较大
            layer_num=32,
            batch_size=16,
            microbatch_size=1,
            sequence_length=2048,
            hidden_size=4096,
            num_attention_heads=32
        )
    elif model_name == "gpt3_13b":
        res = Model(
            name="gpt3_13b",   # GPT-3 超大
            layer_num=40,
            batch_size=8,
            microbatch_size=1,
            sequence_length=2048,
            hidden_size=5120,
            num_attention_heads=40
        )
    elif model_name == "gpt3-175b":
        res = Model(
            name="gpt3-175b",   # GPT-3 Davinci
            layer_num=96,
            batch_size=16,
            microbatch_size=1,
            sequence_length=2048,
            hidden_size=12288,
            num_attention_heads=96
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if DEBUG:
        res.print_info()
    return res
    
def get_device(device_name: str) -> Device:
    # 根据设备名称返回对应的Device对象
    if device_name == "A100-40GB":
        return Device(
            name="A100",
            tflops=312,  # FP16 TFLOPS
            memory_GB=40.0
        )
    elif device_name == "A100-80GB":
        return Device(
            name="A100",
            tflops=312,  # FP16 TFLOPS
            memory_GB=80.0
        )
    elif device_name == "V100-32GB":
        return Device(
            name="V100",
            tflops=125,  # FP16 TFLOPS
            memory_GB=32.0
        )
    elif device_name == "V100-32GB-TP2":
        return Device(
            name="V100",
            tflops=125,  # FP16 TFLOPS
            memory_GB=32.0,
            tp=2
        )
    elif device_name == "V100-16GB":
        return Device(
            name="V100",
            tflops=125,  # FP16 TFLOPS
            memory_GB=16.0
        )
    elif device_name == "RTX3090-24GB":
        return Device(
            name="3090",
            tflops=71,  # FP16 TFLOPS
            memory_GB=24.0
        )
    elif device_name == "RTX4090-24GB":
        return Device(
            name="4090",
            tflops=330,  # FP16 TFLOPS
            memory_GB=24.0
        )
    elif device_name == "RTX5090-32GB":
        return Device(
            name="5090",
            tflops=419,  # FP16 TFLOPS
            memory_GB=32.0
        )
    elif device_name == "H20-96GB":
        return Device(
            name="H20",
            tflops=148,  # FP16 TFLOPS
            memory_GB=96.0
        )
    elif device_name == "H20-96GB-TP2":
        return Device(
            name="H20",
            tflops=148,  # FP16 TFLOPS
            memory_GB=96.0,
            tp=2
        )
    else:
        raise ValueError(f"Unknown device name: {device_name}")