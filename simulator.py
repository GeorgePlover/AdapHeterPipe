from typing import Any, List, Tuple, Dict
from my_profile import Worker, get_model, get_device

# 根据指定逻辑模拟流水线的结果

EPS = 1e-3
MEM_PROTECT = True

class SimConf:
    def __init__(self,
                 stage_cnt: int,
                 microbatch_cnt: int,
                 workers: List[Worker],
                 stages: List[Dict], # {"worker_id":int, "layer_range": Tuple[int, int], "layer_num":int}
                 ):
        self.stage_cnt = stage_cnt
        self.microbatch_cnt = microbatch_cnt
        self.workers = workers
        self.stages = stages
        
        self.worker_cnt = len(workers)
        pass
class Task:
    def __init__(self,
                microbatch_id: int,
                stage_id: int,
                task_type: str,  # "F", "B", "W"
                simconf: SimConf,
                remain_mem: float = 0.0
                ):
        self.stage_id = stage_id
        self.microbatch_id = microbatch_id
        self.task_type = task_type
        self.simconf = simconf
        self.remain_mem = remain_mem  # 该任务执行前需要保留的显存大小
        
        self.worker_id = simconf.stages[stage_id]["worker_id"]
        self.layer_num = simconf.stages[stage_id]["layer_num"]
        self.duration = simconf.workers[self.worker_id].time_per_layer(task_type) * self.layer_num
        self.memory_overhead = simconf.workers[self.worker_id].active_mem_per_layer(task_type) * self.layer_num  # 该任务占用的显存大小
        self.available_time = 0.0  # 该任务可开始的时间点
        self.start_time = None
        self.end_time = None
        self.prev_task = []  # 该任务的前驱任务
        self.succ_task = []  # 该任务的后继任务
        
    def remove_prev_task(self, task: 'Task'):
        return self.prev_task.remove(task)
    
    def release_succ_tasks(self)-> List['Task']: # topological sort
        res = []
        for succ in self.succ_task:
            succ.available_time = max(succ.available_time, self.end_time)
            succ.remove_prev_task(self)
            if len(succ.prev_task) == 0:
                res.append(succ)
        return res
        

class WorkerSim:
    def __init__(self, worker_id: int, simconf: SimConf):
        self.worker_id = worker_id
        self.simconf = simconf
        self.worker = simconf.workers[worker_id]
        self.layer_num = sum(stage["layer_num"] for stage in simconf.stages if stage["worker_id"] == worker_id)
        
        self.total_mem = self.worker.memory_limit()  # 该worker的总内存
        self.static_mem = self.worker.static_mem_per_layer() * self.layer_num  # 该worker的静态内存占用
        self.available_mem = self.total_mem - self.static_mem  # 该worker的可用内存
        
        self.available_time = 0.0  # 该worker下一个可用时间点
        self.task_lists = [] # 该worker的任务列表
        self.time_first_task = None  # 该worker第一个任务的开始时间
        self.time_last_task = 0.0   # 该worker最后一个任务的结束时间
        self.time_total_busy = 0.0  # 该worker总的忙碌时间
        self.peak_mem_usage = 0.0  # 该worker的峰值内存使用量
        
    def execute_next_task(self, another_available_time: float, worker_sims: List['WorkerSim'])-> bool:
        # 对列表里的任务按照microbatch_id排序，找出可执行的任务
        
        self.task_lists.sort(key=lambda task: task.microbatch_id)
        
        first_available_FB_tasks = None
        first_available_W_task = None
        bubble_tasks = []
        for task in self.task_lists:
            if (task.memory_overhead > self.available_mem or 
                (MEM_PROTECT and task.remain_mem > self.available_mem)):
                continue
            if task.available_time <= self.available_time:
                if task.task_type in ["F", "B"]:
                    if first_available_FB_tasks is None:
                        first_available_FB_tasks = task
                elif task.task_type == "W":
                    if first_available_W_task is None:
                        first_available_W_task = task
            else:
                bubble_tasks.append(task)
        bubble_tasks.sort(key=lambda task: task.available_time)
        
        # 决策要执行的任务 TODO: 加入显存逻辑
        if first_available_FB_tasks is not None:
            task = first_available_FB_tasks
        elif first_available_W_task is not None:
            task = first_available_W_task
        elif bubble_tasks:
            if bubble_tasks[0].available_time <= another_available_time:
                task = bubble_tasks[0]
            else:
                if another_available_time < float('inf'):
                    self.available_time = another_available_time
                return False # 没有任务可执行，等待另一个worker
        else:
            if another_available_time < float('inf'):
                self.available_time = another_available_time
            return False # 因为显存限制，没有任务可执行，等待另一个worker
        
        self.available_mem -= task.memory_overhead
        task.start_time = max(self.available_time, task.available_time)
        task.end_time = task.start_time + task.duration
        self.available_time = task.end_time
        self.task_lists.remove(task)
        
        # 释放后继任务
        released_tasks = task.release_succ_tasks()
        for released_task in released_tasks:
            worker_id = released_task.worker_id
            worker_sim = worker_sims[worker_id]
            worker_sim.task_lists.append(released_task)
            
        # 更新worker的统计信息
        if self.time_first_task is None:
            self.time_first_task = task.start_time
        self.time_last_task = task.end_time
        self.time_total_busy += task.duration
        self.peak_mem_usage = max(self.peak_mem_usage, self.worker.memory_limit() - self.available_mem - self.static_mem)
        
        return True # 成功执行了任务
    
    def worker_bubble_rate(self) -> float:
        if self.time_first_task is None:
            return 1.0
        total_time = self.time_last_task - self.time_first_task
        busy_time = self.time_total_busy
        bubble_time = total_time - busy_time
        return bubble_time / total_time
    
    def worker_peak_mem_rate(self) -> float:
        return (self.peak_mem_usage + self.static_mem) / self.worker.memory_limit()
    
class Simulator:
    def __init__(self, config: SimConf):
        self.config = config
        self.tasks_array = self._task_matrix(config) # [microbatch_id][stage_id]["F"/"B"/"W"]->Task
        self.worker_sims = self._worker_sims(config)
    
    def _task_matrix(self, config: SimConf) -> List[List[Dict[str, Task]]]:
        '''
        构建任务矩阵 通过microbatch_id和stage_id还有type索引到对应的任务
        为任务添加前驱任务和后继任务
        '''
        # 用于计算每个F任务后续在同一个worker上的激活内存消耗
        worker_total_layer_active_mem = [0.0 for _ in range(config.worker_cnt)]
        stage_succ_layer_active_mem = [0.0 for _ in range(config.stage_cnt)]
        for stage in range(config.stage_cnt-1, -1, -1):
            worker_id = config.stages[stage]["worker_id"]
            layer_num = config.stages[stage]["layer_num"]
            mem = layer_num * config.workers[worker_id].active_mem_per_layer("F")
            worker_total_layer_active_mem[worker_id] += mem
            stage_succ_layer_active_mem[stage] = worker_total_layer_active_mem[worker_id]
        
        # 构建任务矩阵
        task_matrix = []
        for mb_id in range(config.microbatch_cnt):
            mb_tasks = []
            for stage_id in range(config.stage_cnt):
                type2task = {}
                for task_type in ["F", "B", "W"]:
                    
                    remain_mem = 0.0
                    if task_type == "F":
                        remain_mem = stage_succ_layer_active_mem[stage_id]
                        
                    task = Task(mb_id, stage_id, task_type, config, remain_mem = remain_mem)
                    type2task[task_type] = task
                    # 添加前驱任务和后继任务
                    if task_type == "F":
                        if stage_id > 0:
                            prev_task = mb_tasks[stage_id-1]["F"]
                            task.prev_task.append(prev_task)
                            prev_task.succ_task.append(task)
                        if mb_id > 0:
                            prev_task = task_matrix[mb_id - 1][stage_id]["F"]
                            task.prev_task.append(prev_task)
                            prev_task.succ_task.append(task)
                    elif task_type == "B":
                        if stage_id > 0:
                            succ_task = mb_tasks[stage_id-1]["B"]
                            task.succ_task.append(succ_task)
                            succ_task.prev_task.append(task)
                        if mb_id > 0:
                            prev_task = task_matrix[mb_id - 1][stage_id]["B"]
                            task.prev_task.append(prev_task)
                            prev_task.succ_task.append(task)
                        if stage_id == config.stage_cnt - 1:
                            prev_task = type2task["F"]
                            task.prev_task.append(prev_task)
                            prev_task.succ_task.append(task)
                    elif task_type == "W":
                        if mb_id > 0:
                            prev_task = task_matrix[mb_id - 1][stage_id]["W"]
                            task.prev_task.append(prev_task)
                            prev_task.succ_task.append(task)
                        prev_task = type2task["B"]
                        task.prev_task.append(prev_task)
                        prev_task.succ_task.append(task)
                    
                mb_tasks.append(type2task)
            task_matrix.append(mb_tasks)
        return task_matrix
    
    def _worker_sims(self, config: SimConf) -> List[WorkerSim]:
        '''
        构建每个worker的模拟器
        '''
        worker_sims = []
        for worker_id in range(config.worker_cnt):
            worker_sim = WorkerSim(worker_id, config)
            worker_sims.append(worker_sim)
        
        # 给stage 0的worker分配第一个F任务
        mb_id = 0
        stage_id = 0
        worker_id = config.stages[stage_id]["worker_id"]
        task = self.tasks_array[mb_id][stage_id]["F"]
        worker_sim = worker_sims[worker_id]
        worker_sim.task_lists.append(task)
        
        return worker_sims
        
    def choose_next_workers(self) -> 'WorkerSim':
        '''
        按照任务列表是否有值为第一关键字，available_time从小到大为第二关键字排序
        '''
        res = sorted(
            [worker_sim for worker_sim in self.worker_sims],
            key=lambda ws: (len(ws.task_lists) == 0, ws.available_time)
        )
        return res
    
    def workers_bubble_rate(self) -> List[float]:
        '''
        返回每个worker的bubble rate
        '''
        return [worker_sim.worker_bubble_rate() for worker_sim in self.worker_sims]
    
    def worker_peak_mem_rate(self) -> List[float]:
        '''
        返回每个worker的峰值内存使用量
        '''
        return [worker_sim.worker_peak_mem_rate() for worker_sim in self.worker_sims]

    def run(self):
        # 模拟流水线处理逻辑
        next_worker_sims = self.choose_next_workers()
        iteration = 0
        retry = 0
        while True:
            for idx, worker_sim in enumerate(next_worker_sims):
                if idx == len(next_worker_sims) - 1:
                    another_available_time = float('inf')
                else:
                    another_available_time = next_worker_sims[idx + 1].available_time
                executed = worker_sim.execute_next_task(another_available_time, self.worker_sims)
                if executed:
                    retry = 0
                    break  # 本轮只执行一个任务，执行完后重新选择下一个worker
            if not executed:
                retry += 1
                if retry > self.config.worker_cnt:    
                    assert False, "Too many retry. There might be a deadlock."
            
            next_worker_sims = self.choose_next_workers()
            iteration += 1
            if sum(len(ws.task_lists) for ws in self.worker_sims) == 0:
                break  # 所有任务执行完毕
            
        print(f"Simulation finished in {iteration} iterations.")
        print("Worker bubble rates:", self.workers_bubble_rate())
        print("Worker peak memory usages:", self.worker_peak_mem_rate())
            
    def pipe_res(self):
        # 返回由task_matrix拍扁的task列表
        return [task for mb_tasks in self.tasks_array for stage_tasks in mb_tasks for task in stage_tasks.values()]
    
def test_simulator_zb():
    from visual import generate_gantt_chart
    config = SimConf(
        stage_cnt=4,
        microbatch_cnt=8,
        workers=[Worker(device=None,model=get_model("gpt2-small")), Worker(device=None,model=get_model("gpt2-small")), 
                 Worker(device=None,model=get_model("gpt2-small")), Worker(device=None,model=get_model("gpt2-small"))],
        stages=[
            {"worker_id":0, "layer_range": (0, 4), "layer_num":4},
            {"worker_id":1, "layer_range": (4, 8), "layer_num":4},
            {"worker_id":2, "layer_range": (8, 12), "layer_num":4},
            {"worker_id":3, "layer_range": (12, 16), "layer_num":4},
        ]
    )
    simulator = Simulator(config)
    simulator.run()
    tasks = simulator.pipe_res()
    generate_gantt_chart(tasks, "gantt_chart_zb.png")
    
def test_simulator_zb_v():
    from visual import generate_gantt_chart
    
    device = get_device("A100-80GB")
    
    config = SimConf(
        stage_cnt=8,
        microbatch_cnt=8,
        workers=[Worker(device=device,model=get_model("gpt2-small")), Worker(device=device,model=get_model("gpt2-small")), 
                 Worker(device=device,model=get_model("gpt2-small")), Worker(device=device,model=get_model("gpt2-small"))],
        stages=[
            {"worker_id":0, "layer_range": (0, 2), "layer_num":2},
            {"worker_id":1, "layer_range": (2, 4), "layer_num":2},
            {"worker_id":2, "layer_range": (4, 6), "layer_num":2},
            {"worker_id":3, "layer_range": (6, 8), "layer_num":2},
            {"worker_id":3, "layer_range": (8, 10), "layer_num":2},
            {"worker_id":2, "layer_range": (10, 12), "layer_num":2},
            {"worker_id":1, "layer_range": (12, 14), "layer_num":2},
            {"worker_id":0, "layer_range": (14, 16), "layer_num":2},
        ]
    )
    simulator = Simulator(config)
    simulator.run()
    tasks = simulator.pipe_res()
    generate_gantt_chart(tasks, "gantt_chart_zb_v.png")
    
if __name__ == "__main__":
    test_simulator_zb_v()