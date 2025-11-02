from simulator import Simulator, SimConf, Task
from typing import List, Dict
from my_profile import Model, Worker, get_model, get_device

class Strategy:
    def __init__(self, name: str):
        self.name = name
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        # 根据策略构造流水线阶段划分
        raise NotImplementedError("This method should be overridden by subclasses")

class EvenLayerStrategy(Strategy):
    def __init__(self):
        super().__init__("EvenLayerStrategy")
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stage_cnt = len(workers)
        layers_per_stage = model.layer_num // stage_cnt
        remaining_layers = model.layer_num % stage_cnt
        stages = []
        applied_layers = 0
        for i in range(stage_cnt):
            start_layer = applied_layers
            end_layer = applied_layers + layers_per_stage + (1 if i < remaining_layers else 0)
            stages.append({
                "worker_id": i,
                "layer_range": (start_layer, end_layer),
                "layer_num": end_layer - start_layer
            })
            applied_layers += end_layer - start_layer
        return stages

class EvenVshapeStrategy(Strategy):
    def __init__(self):
        super().__init__("EvenVshapeStrategy")
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        # 构造V形划分的流水线阶段
        stage_cnt = len(workers)
        layers_per_stage = model.layer_num // stage_cnt
        remaining_layers = model.layer_num % stage_cnt
        
        w_layers = [layers_per_stage + (1 if i < remaining_layers else 0) for i in range(stage_cnt)]
        front = []
        back = []
        wids = list(range(stage_cnt))
        for i in range(stage_cnt):
            front.append(w_layers[i] // 2)
            back.append(w_layers[i] - front[-1])
        
        stage_cnt = stage_cnt * 2
        for i in reversed(range(0, stage_cnt//2)):
            front.append(back[i])
            wids.append(i)
        
        stages = []
        applied_layers = 0
        for i in range(stage_cnt):
            start_layer = applied_layers
            end_layer = applied_layers + front[i]
            stages.append({
                "worker_id": wids[i],
                "layer_range": (start_layer, end_layer),
                "layer_num": end_layer - start_layer
            })
            applied_layers += front[i]
        return stages
    
class DivByFlopsStrategy(Strategy):
    def __init__(self):
        super().__init__("DivByFlopsStrategy")
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        total_flops = sum(worker.device.tflops for worker in workers)
        pre_apply = [int(worker.device.tflops / total_flops * model.layer_num) for worker in workers]
        remaining = model.layer_num - sum(pre_apply)
        for i in range(remaining):
            pre_apply[i % len(workers)] += 1
        
        stages = []
        applied_layers = 0
        for i in range(len(workers)):
            layer_num = pre_apply[i]
            start_layer = applied_layers
            end_layer = applied_layers + layer_num
            stages.append({
                "worker_id": i,
                "layer_range": (start_layer, end_layer),
                "layer_num": end_layer - start_layer
            })
            applied_layers += layer_num
        return stages
    
class DivByFlopsVshapeStrategy(Strategy):
    def __init__(self):
        super().__init__("DivByFlopsVshapeStrategy")
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stage_cnt = len(workers)
        total_flops = sum(worker.device.tflops for worker in workers)
        pre_apply = [int(worker.device.tflops / total_flops * model.layer_num) for worker in workers]
        remaining = model.layer_num - sum(pre_apply)
        for i in range(remaining):
            pre_apply[i % len(workers)] += 1
        
        w_layers = pre_apply
        front = []
        back = []
        wids = list(range(stage_cnt))
        for i in range(stage_cnt):
            front.append(w_layers[i] // 2)
            back.append(w_layers[i] - front[-1])
        
        stage_cnt = stage_cnt * 2
        for i in reversed(range(0, stage_cnt//2)):
            front.append(back[i])
            wids.append(i)
        
        stages = []
        applied_layers = 0
        for i in range(stage_cnt):
            start_layer = applied_layers
            end_layer = applied_layers + front[i]
            stages.append({
                "worker_id": wids[i],
                "layer_range": (start_layer, end_layer),
                "layer_num": end_layer - start_layer
            })
            applied_layers += front[i]
        return stages

class DivByMemoryStrategy(Strategy):
    def __init__(self):
        super().__init__("DivByMemoryStrategy")
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        total_mem = sum(worker.device.memory_bytes() for worker in workers)
        pre_apply = [int(worker.device.memory_bytes() / total_mem * model.layer_num) for worker in workers]
        remaining = model.layer_num - sum(pre_apply)
        for i in range(remaining):
            pre_apply[i % len(workers)] += 1
        
        stages = []
        applied_layers = 0
        for i in range(len(workers)):
            layer_num = pre_apply[i]
            start_layer = applied_layers
            end_layer = applied_layers + layer_num
            stages.append({
                "worker_id": i,
                "layer_range": (start_layer, end_layer),
                "layer_num": end_layer - start_layer
            })
            applied_layers += layer_num
        return stages

class DivByMemoryVshapeStrategy(Strategy):
    def __init__(self):
        super().__init__("DivByMemoryVshapeStrategy")
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stage_cnt = len(workers)
        total_mem = sum(worker.device.memory_bytes() for worker in workers)
        pre_apply = [int(worker.device.memory_bytes() / total_mem * model.layer_num) for worker in workers]
        remaining = model.layer_num - sum(pre_apply)
        for i in range(remaining):
            pre_apply[i % len(workers)] += 1
        
        w_layers = pre_apply
        front = []
        back = []
        wids = list(range(stage_cnt))
        for i in range(stage_cnt):
            front.append(w_layers[i] // 2)
            back.append(w_layers[i] - front[-1])
        
        stage_cnt = stage_cnt * 2
        for i in reversed(range(0, stage_cnt//2)):
            front.append(back[i])
            wids.append(i)
        
        stages = []
        applied_layers = 0
        for i in range(stage_cnt):
            start_layer = applied_layers
            end_layer = applied_layers + front[i]
            stages.append({
                "worker_id": wids[i],
                "layer_range": (start_layer, end_layer),
                "layer_num": end_layer - start_layer
            })
            applied_layers += front[i]
        return stages
    
class HandCraftedStrategy(Strategy):
    def __init__(self):
        super().__init__("HandCraftedStrategy")
        
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stages = [
            {"worker_id":0, "layer_range": (0, 2), "layer_num":2},
            {"worker_id":1, "layer_range": (2, 4), "layer_num":2},
            {"worker_id":1, "layer_range": (4, 6), "layer_num":2},
            {"worker_id":0, "layer_range": (6, 8), "layer_num":2},
            {"worker_id":2, "layer_range": (8, 12), "layer_num":4},
            {"worker_id":3, "layer_range": (12, 16), "layer_num":4},
            {"worker_id":3, "layer_range": (16, 20), "layer_num":4},
            {"worker_id":2, "layer_range": (20, 24), "layer_num":4},
        ]
        return stages
    
def test_strategy(model_name:str, 
                  workers_device_names: List[str], 
                  strategy: Strategy,
                  test_name: str):
    model = get_model(model_name)
    worker_cnt = len(workers_device_names)
    workers = []
    
    for i in range(worker_cnt):
        device = get_device(workers_device_names[i])
        workers.append(Worker(device=device, model=model))
    
    stages = strategy.construct_stages(model, workers)
    
    config = SimConf(
        stage_cnt=len(stages),
        microbatch_cnt=model.batch_size // model.microbatch_size,
        workers=workers,
        stages=stages
    )
    
    simulator = Simulator(config)
    simulator.run()
    for worker_sim in simulator.worker_sims:
        print("========================================")
        print(f"Worker ID: {worker_sim.worker_id}")
        print(f"Worker Device: {worker_sim.worker.device.name} Load Layers: {worker_sim.layer_num}")
        print(f"Worker Memory Limit: {worker_sim.worker.memory_limit()/(2**30):.2f} GB")
        print(f"Static Memory: {worker_sim.static_mem/(2**30):.2f} GB")
        print(f"Active Memory Peak Usage: {worker_sim.peak_mem_usage/(2**30):.2f} GB")
        print(f"Worker Memory Peak Rate: {worker_sim.worker_peak_mem_rate():.4f}, Bubble Rate: {worker_sim.worker_bubble_rate():.4f}")
    
    from visual import generate_gantt_chart
    generate_gantt_chart(simulator.pipe_res(), f"{test_name}_gantt_chart.png")
    print("\n\n")

if __name__ == "__main__":
    # 
    test_strategy(
        model_name="gpt3-1.3b",
        workers_device_names=["V100-32GB", "V100-32GB", "RTX4090-24GB", "RTX4090-24GB"],
        strategy=DivByFlopsVshapeStrategy(),
        test_name="Div_By_Flops_Vshape_Strategy_test_1.3B"
    )
    test_strategy(
        model_name="gpt3-1.3b",
        workers_device_names=["V100-32GB", "V100-32GB", "RTX4090-24GB", "RTX4090-24GB"],
        strategy=DivByFlopsStrategy(),
        test_name="Div_By_Flops_Strategy_test_1.3B"
    )
    test_strategy(
        model_name="gpt3-1.3b",
        workers_device_names=["V100-32GB", "V100-32GB", "RTX4090-24GB", "RTX4090-24GB"],
        strategy=DivByMemoryStrategy(),
        test_name="Div_By_Memory_Strategy_test_1.3B"
    )
    test_strategy(
        model_name="gpt3-1.3b",
        workers_device_names=["V100-32GB", "V100-32GB", "RTX4090-24GB", "RTX4090-24GB"],
        strategy=DivByMemoryVshapeStrategy(),
        test_name="Div_By_Memory_Vshape_Strategy_test_1.3B"
    )
    test_strategy(
        model_name="gpt3-1.3b",
        workers_device_names=["V100-32GB", "V100-32GB", "RTX4090-24GB", "RTX4090-24GB"],
        strategy=HandCraftedStrategy(),
        test_name="Hand_Crafted_Strategy_test_1.3B"
    )
    