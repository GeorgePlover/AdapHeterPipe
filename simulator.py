from typing import Any, List, Tuple, Dict
from my_profile import Worker, get_model, get_device
from my_common import DEBUG, MEM_PROTECT
# 根据指定逻辑模拟流水线的结果

class SimConf:
    def __init__(self,
                 stage_cnt: int,
                 microbatch_cnt: int,
                 workers: List[Worker],
                 stages: List[Dict], # {"worker_id":int, "layer_range": Tuple[int, int], "layer_num":int}
                 OMMIT_OOM: bool = False,
                 NO_W: bool = False,
                 ):
        self.stage_cnt = stage_cnt
        self.microbatch_cnt = microbatch_cnt
        self.workers = workers
        self.stages = stages
        self.OMMIT_OOM = OMMIT_OOM
        self.NO_W = NO_W
        
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
        # 处理 WB 合并的情况
        if simconf.NO_W and task_type == "B":
            self.duration += simconf.workers[self.worker_id].time_per_layer("W") * self.layer_num
        
        
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
            if ( (self.simconf.OMMIT_OOM is False) and (task.memory_overhead > self.available_mem or 
                (MEM_PROTECT and task.remain_mem > self.available_mem)) ):
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
        self.STABLE_SCHEDULE = config.OMMIT_OOM  # 是否使用稳定调度策略
        self.NO_W = config.NO_W  # 是否使用W任务
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
                for task_type in (["F", "B", "W"] if self.NO_W is False else ["F", "B"]):
                    
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
    
    def apply_a_succ_has_b(self, a:Task, b:Task):
        a.succ_task.append(b)
        b.prev_task.append(a)
    
    def apply_chain_tasks(self, tasks: List[Task]):
        for i in range(len(tasks)-1):
            self.apply_a_succ_has_b(tasks[i], tasks[i+1])
    
    def use_1f1b_schedule(self):
        '''
            使用1F1B调度策略
        '''
        assert self.STABLE_SCHEDULE is False, "STABLE_SCHEDULE should be False"
        assert self.NO_W is True, "NO_W should be True"
        assert self.config.stage_cnt == self.config.worker_cnt, "stage_cnt should be equal to worker_cnt"
        
        self.STABLE_SCHEDULE = True
        
        config = self.config
        matrix = self.tasks_array # [microbatch_id][stage_id]["F"/"B"/"W"]->Task
        stage_cnt = config.stage_cnt
        mb_cnt = config.microbatch_cnt
        
        for stage_id in range(stage_cnt):
            warmup = min(stage_cnt - stage_id - 1, mb_cnt - 1)
            task_ordered = []
            for mb_id in range(warmup): # Warmup
                task_ordered.append(matrix[mb_id][stage_id]["F"])
            for mb_id in range(warmup, mb_cnt): # 1F1B
                task_ordered.append(matrix[mb_id][stage_id]["F"])
                task_ordered.append(matrix[mb_id - warmup][stage_id]["B"])
            for mb_id in range(mb_cnt - warmup, mb_cnt): # Cooldown
                task_ordered.append(matrix[mb_id][stage_id]["B"])
            self.apply_chain_tasks(task_ordered)
                
    def use_interleaved_1f1b_schedule(self, interleaved_degree: int):
        '''
            使用交错1F1B调度策略
        '''
        assert self.STABLE_SCHEDULE is False, "STABLE_SCHEDULE should be False"
        assert self.NO_W is True, "NO_W should be True"
        assert self.config.worker_cnt * interleaved_degree == self.config.stage_cnt, "stage_cnt should be equal to worker_cnt * vp"
        assert self.config.microbatch_cnt % self.config.worker_cnt == 0, "microbatch_cnt should be divisible by worker_cnt"
        
        self.STABLE_SCHEDULE = True
        
        config = self.config
        matrix = self.tasks_array # [microbatch_id][stage_id]["F"/"B"/"W"]->Task
        stage_cnt = config.stage_cnt
        worker_cnt = config.worker_cnt
        mb_cnt = config.microbatch_cnt
        vp_cnt = interleaved_degree
        
        for worker_id in range(worker_cnt):
            Flist = []
            Blist = []
            for chunk_id in range(mb_cnt // worker_cnt):
                for stage_id in range(worker_id, stage_cnt, worker_cnt):
                    for mb_id in range(chunk_id * worker_cnt, (chunk_id + 1) * worker_cnt):
                        Flist.append(matrix[mb_id][stage_id]["F"])
                for stage_id in range(worker_id+worker_cnt*(vp_cnt-1), -1, -worker_cnt):
                    for mb_id in range(chunk_id * worker_cnt, (chunk_id + 1) * worker_cnt):
                        Blist.append(matrix[mb_id][stage_id]["B"])
                        
            warmup = min((vp_cnt - 1) * worker_cnt + 2 * (worker_cnt - worker_id - 1), mb_cnt * vp_cnt - 1)
            task_ordered = []
            
            for f_id in range(warmup): # Warmup
                task_ordered.append(Flist[f_id])
            for f_id in range(warmup, len(Flist)): # Interleaved 1F1B
                task_ordered.append(Flist[f_id])
                task_ordered.append(Blist[f_id - warmup])
            for b_id in range(len(Blist) - warmup, len(Blist)): # Cooldown
                task_ordered.append(Blist[b_id])
            self.apply_chain_tasks(task_ordered)
            
    def use_zb_schedule(self):
        '''
            使用ZB调度策略
        '''
        assert self.STABLE_SCHEDULE is False, "STABLE_SCHEDULE should be False"
        assert self.NO_W is False, "NO_W should be False"
        assert self.config.stage_cnt == self.config.worker_cnt, "stage_cnt should be equal to worker_cnt"
        
        self.STABLE_SCHEDULE = True
        
        config = self.config
        matrix = self.tasks_array # [microbatch_id][stage_id]["F"/"B"/"W"]->Task
        stage_cnt = config.stage_cnt
        mb_cnt = config.microbatch_cnt
        
        for stage_id in range(stage_cnt):
            warmup_f = min((stage_cnt - stage_id - 1) * 2, mb_cnt - 1)
            warmup_b = min(stage_id * 2, mb_cnt - 1)
            f_id, b_id, w_id = 0, 0, 0
            task_ordered = []
            
            while f_id < mb_cnt or b_id < mb_cnt or w_id < mb_cnt:
                if f_id < mb_cnt:
                    task_ordered.append(matrix[f_id][stage_id]["F"])
                    f_id += 1
                if b_id < mb_cnt and f_id > warmup_f:
                    task_ordered.append(matrix[b_id][stage_id]["B"])
                    b_id += 1
                if w_id < mb_cnt and b_id > warmup_b:
                    task_ordered.append(matrix[w_id][stage_id]["W"])
                    w_id += 1
            
            self.apply_chain_tasks(task_ordered)
            
    def use_zv_vshape_schedule(self):
        '''
            使用ZB-Vshape调度策略
        '''
        assert self.STABLE_SCHEDULE is False, "STABLE_SCHEDULE should be False"
        assert self.NO_W is False, "NO_W should be False"
        assert self.config.stage_cnt == self.config.worker_cnt * 2, "stage_cnt should be equal to worker_cnt * 2"
        
        self.STABLE_SCHEDULE = True
        
        config = self.config
        matrix = self.tasks_array # [microbatch_id][stage_id]["F"/"B"/"W"]->Task
        stage_cnt = config.stage_cnt
        worker_cnt = config.worker_cnt
        mb_cnt = config.microbatch_cnt
        
        for worker_id in range(worker_cnt):
            warmup_front_half = worker_cnt - worker_id - 1
            warmup_front_back_inter = worker_id + 1
            
            f_0 = [matrix[batch_id][worker_id]["F"] for batch_id in range(mb_cnt)]
            f_1 = [matrix[batch_id][stage_cnt-1-worker_id]["F"] for batch_id in range(mb_cnt)]
            b_0 = [matrix[batch_id][worker_id]["B"] for batch_id in range(mb_cnt)]
            b_1 = [matrix[batch_id][stage_cnt-1-worker_id]["B"] for batch_id in range(mb_cnt)]
            w_0 = [matrix[batch_id][worker_id]["W"] for batch_id in range(mb_cnt)]
            w_1 = [matrix[batch_id][stage_cnt-1-worker_id]["W"] for batch_id in range(mb_cnt)]
            
            task_ordered = []
            for _ in range(warmup_front_half):
                for rep in range(2):
                    if len(f_0) > 0:
                        task_ordered.append(f_0.pop(0))
            for _ in range(warmup_front_back_inter):
                if len(f_0) > 0:
                    task_ordered.append(f_0.pop(0))
                if len(f_1) > 0:
                    task_ordered.append(f_1.pop(0))
            for _ in range(warmup_front_half):
                if len(b_1) > 0:
                    task_ordered.append(b_1.pop(0))
                if len(w_1) > 0:
                    task_ordered.append(w_1.pop(0))
                if len(f_1) > 0:
                    task_ordered.append(f_1.pop(0))
            while max(len(f_0), len(f_1)) > 0:
                if len(b_1) > 0:
                    task_ordered.append(b_1.pop(0))
                if len(b_0) > 0:
                    task_ordered.append(b_0.pop(0))
                if len(w_1) > 0:
                    task_ordered.append(w_1.pop(0))
                if len(w_0) > 0:
                    task_ordered.append(w_0.pop(0))
                if len(f_0) > 0:
                    task_ordered.append(f_0.pop(0))
                if len(f_1) > 0:
                    task_ordered.append(f_1.pop(0))
            for _ in range(warmup_front_back_inter):
                if len(b_1) > 0:
                    task_ordered.append(b_1.pop(0))
                if len(b_0) > 0:
                    task_ordered.append(b_0.pop(0))
            while max(len(b_0), len(w_0), len(w_1)) > 0:
                if len(w_1) > 0:
                    task_ordered.append(w_1.pop(0))
                elif len(w_0) > 0:
                    task_ordered.append(w_0.pop(0))
                if len(b_1) > 0:
                    task_ordered.append(b_1.pop(0))
                elif len(b_0) > 0:
                    task_ordered.append(b_0.pop(0))
                if len(w_0) > 0:
                    task_ordered.append(w_0.pop(0))
                if len(b_0) > 0:
                    task_ordered.append(b_0.pop(0))
                
                
            
            self.apply_chain_tasks(task_ordered)
    
    
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
                    raise Exception(f"iteration: {iteration} - OOM.")
                    # assert False, "Too many retry. There might be a deadlock."
            
            next_worker_sims = self.choose_next_workers()
            iteration += 1
            if sum(len(ws.task_lists) for ws in self.worker_sims) == 0:
                break  # 所有任务执行完毕
        if DEBUG:
            print(f"Simulation finished in {iteration} iterations.")
            print("Worker bubble rates:", self.workers_bubble_rate())
            print("Worker peak memory usages:", self.worker_peak_mem_rate())
            
    def pipe_res(self):
        # 返回由task_matrix拍扁的task列表
        return [task for mb_tasks in self.tasks_array for stage_tasks in mb_tasks for task in stage_tasks.values()]
    
    def pipe_e2e_time(self) -> float:
        # 返回流水线的端到端时间
        return max(worker_sim.time_last_task for worker_sim in self.worker_sims)
    
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