from typing import Any, List, Tuple, Dict
from my_profile import Worker

# 根据指定逻辑模拟流水线的结果

EPS = 1e-6

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
                    simconf: SimConf
                    ):
        self.stage_id = stage_id
        self.microbatch_id = microbatch_id
        self.task_type = task_type
        self.simconf = simconf
        
        self.worker_id = simconf.stages[stage_id]["worker_id"]
        self.layer_num = simconf.stages[stage_id]["layer_num"]
        self.duration = simconf.workers[self.worker_id].time_per_layer(task_type) * self.layer_num
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
        
        self.available_time = 0.0  # 该worker下一个可用时间点
        self.task_lists = [] # 该worker的任务列表
        
    def execute_next_task(self, another_available_time: float, worker_sims: List['WorkerSim'])-> bool:
        # 对列表里的任务按照microbatch_id排序，找出可执行的任务
        self.task_lists.sort(key=lambda task: task.microbatch_id)
        
        first_available_FB_tasks = None
        first_available_W_task = None
        bubble_tasks = []
        for task in self.task_lists:
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
                self.available_time = another_available_time + EPS
                return False # 没有任务可执行，等待另一个worker
        else:
            assert False, "Should not happen. No available task to execute."
        
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
        
        return True # 成功执行了任务
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
        task_matrix = []
        for mb_id in range(config.microbatch_cnt):
            mb_tasks = []
            for stage_id in range(config.stage_cnt):
                type2task = {}
                for task_type in ["F", "B", "W"]:
                    task = Task(mb_id, stage_id, task_type, config)
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
        选择还有任务的workers，按照available_time最小的worker优先排序
        '''
        res = sorted(
            [worker_sim for worker_sim in self.worker_sims if len(worker_sim.task_lists) > 0],
            key=lambda ws: ws.available_time
        )
        return res

    def run(self):
        # 模拟流水线处理逻辑
        next_worker_sims = self.choose_next_workers()
        iteration = 0
        while next_worker_sims:
            if len(next_worker_sims) == 1:
                another_available_time = float('inf')
            else:
                another_available_time = next_worker_sims[1].available_time
            worker_sim = next_worker_sims[0]
            executed = worker_sim.execute_next_task(another_available_time, self.worker_sims)
            next_worker_sims = self.choose_next_workers()
            iteration += 1
            
        print(f"Simulation finished in {iteration} iterations.")
            
    def pipe_res(self):
        # 返回由task_matrix拍扁的task列表
        return [task for mb_tasks in self.tasks_array for stage_tasks in mb_tasks for task in stage_tasks.values()]
    
def test_simulator_zb():
    from visual import generate_gantt_chart
    config = SimConf(
        stage_cnt=4,
        microbatch_cnt=8,
        workers=[Worker(device_type = 0), Worker(device_type = 0), Worker(device_type = 0), Worker(device_type = 0)],
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
    config = SimConf(
        stage_cnt=8,
        microbatch_cnt=8,
        workers=[Worker(device_type = 0), Worker(device_type = 0), Worker(device_type = 0), Worker(device_type = 0)],
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
    test_simulator_zb()
    test_simulator_zb_v()