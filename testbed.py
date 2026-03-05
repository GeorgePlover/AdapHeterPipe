from simulator import Simulator, SimConf, Task
from typing import List, Dict, Tuple
from my_profile import Model, Worker, get_model, get_device
from SA import SimulatedAnnealer
from my_common import DEBUG, VISUALIZE
import time
import random
import json

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

class EvenLayerStrategyInterleaved(EvenLayerStrategy):
    def __init__(self):
        super().__init__()
        self.name = "EvenLayerStrategyInterleaved"
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stages = super().construct_stages(model, workers)
        interleaved_stages1 = []
        interleaved_stages2 = []
        for stage in stages:
            layer_num = stage["layer_num"]
            mid = layer_num // 2
            interleaved_stages1.append({
                "worker_id": stage["worker_id"],
                "layer_num": mid
            })
            interleaved_stages2.append({
                "worker_id": stage["worker_id"],
                "layer_num": layer_num - mid
            })
        interleaved_stages = interleaved_stages1 + interleaved_stages2
        layer_sum = 0
        for stage_id in range(len(interleaved_stages)):
            stage = interleaved_stages[stage_id]
            layer_sum += stage["layer_num"]
            stage["layer_range"] = (layer_sum - stage["layer_num"], layer_sum)
        return interleaved_stages

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
    
class DivByFlopsStrategyInterleaved(DivByFlopsStrategy):
    def __init__(self):
        super().__init__()
        self.name = "DivByFlopsStrategyInterleaved"
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stages = super().construct_stages(model, workers)
        interleaved_stages1 = []
        interleaved_stages2 = []
        for stage in stages:
            layer_num = stage["layer_num"]
            mid = layer_num // 2
            interleaved_stages1.append({
                "worker_id": stage["worker_id"],
                "layer_num": mid
            })
            interleaved_stages2.append({
                "worker_id": stage["worker_id"],
                "layer_num": layer_num - mid
            })
        interleaved_stages = interleaved_stages1 + interleaved_stages2
        layer_sum = 0
        for stage_id in range(len(interleaved_stages)):
            stage = interleaved_stages[stage_id]
            layer_sum += stage["layer_num"]
            stage["layer_range"] = (layer_sum - stage["layer_num"], layer_sum)
        return interleaved_stages
    
class DivByFlopsVshapeStrategy(Strategy):
    def __init__(self):
        super().__init__("DivByFlopsVshapeStrategy")
    
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stage_cnt = len(workers)
        total_flops = sum(worker.device.tflops for worker in workers)
        pre_apply = [int(worker.device.tflops / total_flops * model.layer_num) for worker in workers]
        
        if all([worker.exist_profiling() for worker in workers]):
            # 存在设备性能数据，按性能比例划分
            total_flops = sum(1.0/worker.get_profiling("forward_time_per_layer") for worker in workers)
            pre_apply = [int(1.0/worker.get_profiling("forward_time_per_layer") / total_flops * model.layer_num) for worker in workers]
            # pre_apply = [8,8,4,4]
        
        remaining = model.layer_num - sum(pre_apply)
        for i in range(remaining):
            pre_apply[i-1-(i % len(workers))] += 1
        
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
    
class DivByMemoryStrategyInterleaved(DivByMemoryStrategy):
    def __init__(self):
        super().__init__()
        self.name = "DivByMemoryStrategyInterleaved"
        
    def construct_stages(self, model:Model, workers: List[Worker]) -> List[Dict]:
        stages = super().construct_stages(model, workers)
        interleaved_stages1 = []
        interleaved_stages2 = []
        for stage in stages:
            layer_num = stage["layer_num"]
            mid = layer_num // 2
            interleaved_stages1.append({
                "worker_id": stage["worker_id"],
                "layer_num": mid
            })
            interleaved_stages2.append({
                "worker_id": stage["worker_id"],
                "layer_num": layer_num - mid
            })
        interleaved_stages = interleaved_stages1 + interleaved_stages2
        layer_sum = 0
        for stage_id in range(len(interleaved_stages)):
            stage = interleaved_stages[stage_id]
            layer_sum += stage["layer_num"]
            stage["layer_range"] = (layer_sum - stage["layer_num"], layer_sum)
        return interleaved_stages

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
            {"worker_id":0, "layer_range": (2, 4), "layer_num":2},
            {"worker_id":1, "layer_range": (4, 6), "layer_num":2},
            {"worker_id":1, "layer_range": (6, 8), "layer_num":2},
            {"worker_id":2, "layer_range": (8, 12), "layer_num":4},
            {"worker_id":2, "layer_range": (12, 16), "layer_num":4},
            {"worker_id":3, "layer_range": (16, 20), "layer_num":4},
            {"worker_id":3, "layer_range": (20, 24), "layer_num":4},
        ]
        return stages

class SAState:
    def __init__(self, list_of_layers_workerid: List[Tuple[int, int]]):
        self.state = list_of_layers_workerid
        
    def from_stages(self, stages: List[Dict]):
        self.state = [(stage["layer_range"][1] - stage["layer_range"][0], stage["worker_id"]) for stage in stages]
        
    def swap(self, i: int, j: int):
        self.state[i], self.state[j] = self.state[j], self.state[i]
        return True
        
    def swap_a_layer(self, i: int, j: int):
        if self.state[i][0] == 1:
            return False
        self.state[i] = (self.state[i][0] - 1, self.state[i][1])
        self.state[j] = (self.state[j][0] + 1, self.state[j][1])
        return True
    
    def to_stages(self) -> List[Dict]:
        stages = []
        worker_id = 0
        layer_start = 0
        for layer_num, worker_id in self.state:
            stages.append({
                "worker_id": worker_id,
                "layer_range": (layer_start, layer_start + layer_num),
                "layer_num": layer_num
            })
            layer_start += layer_num
        return stages
    
    def to_json(self):
        return self.state
class SAOptimizer(SimulatedAnnealer):
    def __init__(self, model_name:str, workers_device_names: List[str],
                 x0: SAState, T0 = 100, Tmin = 0.001, max_iter = 10000, alpha = 0.95, max_stay = 500, seed = 42, swap_color_rate = 0.5):
        super().__init__(x0, T0, Tmin, max_iter, alpha, max_stay, seed)
        self.model_name = model_name
        self.workers_device_names = workers_device_names
        self.model = get_model(model_name)
        self.worker_cnt = len(workers_device_names)
        self.workers = []
        self.swap_color_rate = swap_color_rate
        for i in range(self.worker_cnt):
            device = get_device(workers_device_names[i])
            self.workers.append(Worker(device=device, model=self.model))
        
    
    def energy(self, state: SAState) -> float:
        stages = state.to_stages()
        config = SimConf(
            stage_cnt=len(stages),
            microbatch_cnt=self.model.batch_size // self.model.microbatch_size,
            workers=self.workers,
            stages=stages
        )
        simulator = Simulator(config)
        try:
            simulator.run()
        except Exception as e:
            # 获取OOM情况错误字符串
            if "OOM" in str(e):
                iteration = str(e).split()[1]
                return float(1e9 - int(iteration))
            return float("inf")
        
        return simulator.pipe_e2e_time()
    
    def neighbor(self, state: SAState):
        new_state = SAState(state.state.copy())
        i,j = 1,1
        while i==j:
            i, j = random.sample(range(len(new_state.state)), 2)
            
        if random.random() < self.swap_color_rate:
            new_state.swap(i, j)
        else:
            new_state.swap_a_layer(i, j)
            
        return new_state
    
def test_strategy(model_name:str, 
                  workers_device_names: List[str], 
                  strategy: Strategy = None,
                  test_name: str = "test",
                  stages = None, 
                  pipe_schedule_type: str = "adaptive") -> Tuple[float, Dict]:
    model = get_model(model_name)
    worker_cnt = len(workers_device_names)
    workers = []
    
    for i in range(worker_cnt):
        device = get_device(workers_device_names[i])
        workers.append(Worker(device=device, model=model))
    
    if strategy is not None: 
        stages = strategy.construct_stages(model, workers)
        print(stages)
        
    assert stages is not None, "No strategy is provided and no stages are provided"
    
    if pipe_schedule_type in ["1F1B","Interleaved_1F1B","Gpipe"]:
        no_w = True
    else:
        no_w = False
    
    config = SimConf(
        stage_cnt=len(stages),
        microbatch_cnt=model.batch_size // model.microbatch_size,
        workers=workers,
        stages=stages,
        NO_W=no_w
    )
    
    simulator = Simulator(config)
    if pipe_schedule_type == "1F1B":
        simulator.use_1f1b_schedule()
    elif pipe_schedule_type == "Interleaved_1F1B":
        simulator.use_interleaved_1f1b_schedule(interleaved_degree=2)
    elif pipe_schedule_type == "ZB":
        simulator.use_zb_schedule()
    elif pipe_schedule_type == "ZB_V":
        simulator.use_zv_vshape_schedule()
    elif pipe_schedule_type == "Gpipe":
        simulator.use_gpipe_schedule()
    
    simulator.run()
    workersim_record = simulator.workers_record_res()
    
    if DEBUG:
        for worker_sim in simulator.worker_sims:
            print("========================================")
            print(f"Worker ID: {worker_sim.worker_id}")
            print(f"Worker Device: {worker_sim.worker.device.name} Load Layers: {worker_sim.layer_num}")
            print(f"Worker Memory Limit: {worker_sim.worker.memory_limit()/(2**30):.2f} GB")
            print(f"Static Memory: {worker_sim.static_mem/(2**30):.2f} GB")
            print(f"Active Memory Peak Usage: {worker_sim.peak_mem_usage/(2**30):.2f} GB")
            print(f"Worker Memory Peak Rate: {worker_sim.worker_peak_mem_rate():.4f}, Bubble Rate: {worker_sim.worker_bubble_rate():.4f}")
        print(f"E2E time: {simulator.pipe_e2e_time()} sec\n\n")    
        
    
    if VISUALIZE:
        from visual import generate_gantt_chart
        generate_gantt_chart(simulator.pipe_res(), f"{test_name}_gantt_chart.pdf", no_W=no_w)

    return simulator.pipe_e2e_time(), workersim_record
        
def test_SA(model_name:str, workers_device_names: List[str], test_name: str = "SA_test", pipeline_type: str = "adaptive", T0 = 100, Tmin = 0.001, max_iter = 10000, alpha = 0.95, max_stay = 500, seed = 42, swap_color_rate = 0.5):
    model = get_model(model_name)
    worker_cnt = len(workers_device_names)
    workers = []
    for i in range(worker_cnt):
        device = get_device(workers_device_names[i])
        workers.append(Worker(device=device, model=model))
    
    init_state = SAState([(0,0)])
    init_state.from_stages(DivByMemoryVshapeStrategy().construct_stages(model, workers))   # FIXME: 这里的策略起点需要根据实际情况进行调整
    swap_color_rate = 0.5
    if pipeline_type != "adaptive":
        swap_color_rate = 0.0
    optimizer = SAOptimizer(model_name, workers_device_names, init_state, T0, Tmin, max_iter, alpha, max_stay, seed, swap_color_rate)
    res = optimizer.run()
    print(res["best_state"],res["best_energy"])
    json.dump(res, open(f"{test_name}_SA_result.json", "w"))
    
    record = None
    
    if VISUALIZE:
        e2e_time, record = test_strategy(model_name, workers_device_names, stages=SAState(res["best_state"]).to_stages(), test_name=test_name)
    return res["best_energy"], record
    

def test_gpipe(test_name = "GPipe_test_1.3B"):
    device_name_list = ["A100-80GB", "A100-80GB", "A100-80GB", "A100-80GB"]
    model_name = "gpt3_1.3b"
    test_strategy(
        model_name=model_name,
        workers_device_names=device_name_list,
        strategy=DivByFlopsStrategy(),
        test_name=test_name,
        pipe_schedule_type="Gpipe"
    )

def test_normal_1f1b(test_name = "1F1B_test_1.3B"):
    device_name_list = ["A100-80GB", "A100-80GB", "A100-80GB", "A100-80GB"]
    model_name = "gpt3_1.3b"
    test_strategy(
        model_name=model_name,
        workers_device_names=device_name_list,
        strategy=DivByFlopsStrategy(),
        test_name=test_name,
        pipe_schedule_type="1F1B"
    )

def test_normal_interleaved_1f1b(test_name = "Interleaved_1F1B_test_1.3B"):
    device_name_list = ["A100-80GB", "A100-80GB", "A100-80GB", "A100-80GB"]
    model_name = "gpt3_1.3b"
    test_strategy(
        model_name=model_name,
        workers_device_names=device_name_list,
        strategy=EvenLayerStrategyInterleaved(),
        test_name=test_name,
        pipe_schedule_type="Interleaved_1F1B"
    )
    
def test_normal_zb(test_name = "ZB_test_1.3B"):
    device_name_list = ["A100-80GB", "A100-80GB", "A100-80GB", "A100-80GB"]
    model_name = "gpt3_1.3b"
    test_strategy(
        model_name=model_name,
        workers_device_names=device_name_list,
        strategy=EvenLayerStrategy(),
        test_name=test_name,
        pipe_schedule_type="ZB"
    )
def test_normal_zb_vshape(test_name = "ZB_Vshape_test_1.3B"):
    device_name_list = ["A100-80GB", "A100-80GB", "A100-80GB", "A100-80GB"]
    model_name = "gpt3_1.3b"
    test_strategy(
        model_name=model_name,
        workers_device_names=device_name_list,
        strategy=EvenVshapeStrategy(),
        test_name=test_name,
        pipe_schedule_type="ZB_V"
    )
    
def run_exp(device_name_list: List[str], model_name: str, folder_name: str):
    methods = [
        # {
        #     "name": "1F1B(Even)",
        #     "strategy": EvenLayerStrategy(),
        #     "pipe_schedule_type": "1F1B"
        # },
        # {
        #     "name": "1F1B(DivByFlops)",
        #     "strategy": DivByFlopsStrategy(),
        #     "pipe_schedule_type": "1F1B"
        # },
        # {
        #     "name": "1F1B(DivByMemory)",
        #     "strategy": DivByMemoryStrategy(),
        #     "pipe_schedule_type": "1F1B"
        # },
        # {
        #     "name": "Interleaved_1F1B(Even)",
        #     "strategy": EvenLayerStrategyInterleaved(),
        #     "pipe_schedule_type": "Interleaved_1F1B"
        # },
        # {
        #     "name": "Interleaved_1F1B(DivByFlops)",
        #     "strategy": DivByFlopsStrategyInterleaved(),
        #     "pipe_schedule_type": "Interleaved_1F1B"
        # },
        # {
        #     "name": "Interleaved_1F1B(DivByMemory)",
        #     "strategy": DivByMemoryStrategyInterleaved(),
        #     "pipe_schedule_type": "Interleaved_1F1B"
        # },
        # {
        #     "name": "ZB(Even)",
        #     "strategy": EvenLayerStrategy(),
        #     "pipe_schedule_type": "ZB"
        # },
        # {
        #     "name": "ZB(DivByFlops)",
        #     "strategy": DivByFlopsStrategy(),
        #     "pipe_schedule_type": "ZB"
        # },
        # {
        #     "name": "ZB(DivByMemory)",
        #     "strategy": DivByMemoryStrategy(),
        #     "pipe_schedule_type": "ZB"
        # },
        {
            "name": "ZB-Vshape(Even)",
            "strategy": EvenVshapeStrategy(),
            "pipe_schedule_type": "ZB_V"
        },
        {
            "name": "ZB-Vshape(DivByFlops)",
            "strategy": DivByFlopsVshapeStrategy(),
            "pipe_schedule_type": "ZB_V"
        },
        {
            "name": "ZB-Vshape(DivByMemory)",
            "strategy": DivByMemoryVshapeStrategy(),
            "pipe_schedule_type": "ZB_V"
        },
        # {
        #     "name": "ZB-V-adaptive(Even)",
        #     "strategy": EvenVshapeStrategy(),
        #     "pipe_schedule_type": "adaptive"
        # },
        # {
        #     "name": "ZB-V-adaptive(DivByFlops)",
        #     "strategy": DivByFlopsVshapeStrategy(),
        #     "pipe_schedule_type": "adaptive"
        # },
        # {
        #     "name": "ZB-V-adaptive(DivByMemory)",
        #     "strategy": DivByMemoryVshapeStrategy(),
        #     "pipe_schedule_type": "adaptive"
        # },
        {
            "name": "SA",
            "strategy": None,
            "pipe_schedule_type": "adaptive"
        },
        # {
        #     "name": "SA-wo-adaptive",
        #     "strategy": None,
        #     "pipe_schedule_type": "ZB_V"
        # }
    ]
    res = []
    for method in methods:
        if method["name"] == "SA":
            e2e_time, record = test_SA(
                model_name=model_name,
                workers_device_names=device_name_list,
                test_name=f"{folder_name}/{method['name']}_test_{model_name.replace('.','_')}"
            )
            res.append({
                "name": method["name"],
                "e2e_time": e2e_time,
                "record": record
            })
        elif method["name"] == "SA-wo-adaptive":
            e2e_time, record = test_SA(
                model_name=model_name,
                workers_device_names=device_name_list,
                test_name=f"{folder_name}/{method['name']}_test_{model_name.replace('.','_')}",
                pipeline_type = "ZB_V"
            )
            res.append({
                "name": method["name"],
                "e2e_time": e2e_time,
                "record": record
            })
        else:
            try:
                e2e_time, record = test_strategy(
                    model_name=model_name,
                    workers_device_names=device_name_list,
                    strategy=method["strategy"],
                    test_name=f"{folder_name}/{method['name']}_test_{model_name.replace('.','_')}",
                    pipe_schedule_type=method["pipe_schedule_type"]
                )
                print(f"Method: {method['name']}, E2E Time: {e2e_time} sec")
                res.append({
                    "name": method["name"],
                    "e2e_time": e2e_time,
                    "record": record
                })
            except Exception as e:
                print(f"Method: {method['name']} failed with error: {e}")
                res.append({
                    "name": method["name"],
                    "e2e_time": None,
                    "error": str(e)
                })
    
    json.dump(res, open(f"{folder_name}/result_{model_name.replace('.','_')}.json", "w"))
    
    if VISUALIZE:
        model = get_model(model_name)
        to_throughput = lambda e2e_time: (model.batch_size * model.sequence_length / e2e_time * 1) if e2e_time is not None else 0.0
        result_dict = {r["name"]: to_throughput(r["e2e_time"]) for r in res}
        from visual import generate_result_bar_chart
        generate_result_bar_chart(result_dict, f"{folder_name}/result_bar_chart_{model_name.replace('.','_')}.pdf")
        for item in result_dict:
            print(f"Method: {item}, Throughput: {(result_dict[item]/1000):.2f} K tokens/sec")
            
        from visual import plot_pipeline_metrics
        plot_pipeline_metrics(res, f"{folder_name}/pipeline_metrics_{model_name.replace('.','_')}.pdf")

def create_folder(folder_name: str):
    import os
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

if __name__ == "__main__":
    # test_gpipe("draw/GPipe_draw")
    # test_normal_1f1b("draw/1F1B_draw")
    # test_normal_interleaved_1f1b("draw/Interleaved_1F1B_draw")
    # test_normal_zb("draw/ZB_draw")
    # test_normal_zb_vshape("draw/ZB_Vshape_draw")
    
    # test_normal_zb_vshape()
    
    # device_name_list = [{"device_name":"H20-96GB-TP2", "node_id":0},
    #                     {"device_name":"H20-96GB-TP2", "node_id":0},
    #                     {"device_name":"V100-32GB-TP2", "node_id":1},
    #                     {"device_name":"V100-32GB-TP2", "node_id":1}]
    # model_name = "gpt3_13b"
    # create_folder("hhvv_tp2_13B_results")
    # run_exp(device_name_list, model_name, folder_name="hhvv_tp2_13B_results")
    
    # device_name_list = ["H20-96GB-TP2", "H20-96GB-TP2", "V100-32GB-TP2", "V100-32GB-TP2"]
    # model_name = "gpt3_1.3b"
    # create_folder("hhvv_tp2_1_3B_results")
    # run_exp(device_name_list, model_name, folder_name="hhvv_tp2_1_3B_results")
    
    # -----   H20 * 2 + 5090 * 2
    
    device_name_list = [{"device_name":"H20-96GB", "node_id":0},
                        {"device_name":"H20-96GB", "node_id":0},
                        {"device_name":"RTX5090-32GB", "node_id":1},
                        {"device_name":"RTX5090-32GB", "node_id":1}]
    
    # model_name = "gpt3_760m"
    # create_folder("bw_hh55_760m_results")
    # run_exp(device_name_list, model_name, folder_name="bw_hh55_760m_results")
    
    # model_name = "gpt3_6.7b"
    # create_folder("bw_hh55_6_7b_results")
    # run_exp(device_name_list, model_name, folder_name="bw_hh55_6_7b_results")
    
    model_name = "gpt3_1.3b"
    create_folder("bw_hh55_1_3b_results")
    run_exp(device_name_list, model_name, folder_name="bw_hh55_1_3b_results")
    
    # -----
    
    # device_name_list = ["V100-32GB", "V100-32GB", "RTX4090-24GB", "RTX4090-24GB"]
    # model_name = "gpt3_1.3b"
    # create_folder("vv44_results")
    # run_exp(device_name_list, model_name, folder_name="vv44_results")
    
    # device_name_list = ["V100-32GB", "H20-96GB", "RTX4090-24GB", "RTX5090-32GB"]
    # model_name = "gpt3_760m"
    # create_folder("vh45_760m_results")
    # run_exp(device_name_list, model_name, folder_name="vh45_760m_results")
    
    # device_name_list = ["V100-32GB", "H20-96GB", "RTX4090-24GB", "RTX5090-32GB"]
    # model_name = "gpt3_1.3b"
    # create_folder("vh45_results")
    # run_exp(device_name_list, model_name, folder_name="vh45_results")
    
    # OOM test
    # device_name_list = ["V100-32GB", "H20-96GB", "RTX4090-24GB", "RTX5090-32GB"]
    # model_name = "gpt3_6.7b"
    # create_folder("vh45_oom_results")
    # run_exp(device_name_list, model_name, folder_name="vh45_oom_results")
    
    # ----- large scale
    
    
    # device_name_list = [{"device_name":"H20-96GB-TP8", "node_id":0},
    #                     {"device_name":"H20-96GB-TP8", "node_id":1},
    #                     {"device_name":"H20-96GB-TP8", "node_id":2},
    #                     {"device_name":"H20-96GB-TP8", "node_id":3},
    #                     {"device_name":"RTX5090-32GB-TP8", "node_id":4},
    #                     {"device_name":"RTX5090-32GB-TP8", "node_id":5},
    #                     {"device_name":"RTX5090-32GB-TP8", "node_id":6},
    #                     {"device_name":"RTX5090-32GB-TP8", "node_id":7}]
    
    # model_name = "gpt3_175b-tune"
    # create_folder("bw_hhhh5555_88b_results")
    # run_exp(device_name_list, model_name, folder_name="bw_hhhh5555_88b_results")