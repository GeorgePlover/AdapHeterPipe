import math
import random
from typing import Any, Callable, Optional, Dict


class SimulatedAnnealer:
    """
    可拓展的模拟退火基类。

    使用方式：
    1. 继承本类
    2. 重写 energy(state) 和 neighbor(state)，必要时重写 acceptance/cool/stop
    3. 调用 run() 执行搜索
    """

    def __init__(
        self,
        x0: Any,
        T0: float = 100.0,
        Tmin: float = 1e-3,
        max_iter: int = 10_000,
        alpha: float = 0.95,
        max_stay: int = 500,
        seed: Optional[int] = 42,
    ):
        """
        :param x0: 初始状态（可以是标量、向量、列表、自定义对象等）
        :param T0: 初始温度
        :param Tmin: 最低温度
        :param max_iter: 最大迭代次数
        :param alpha: 默认的几何降温系数
        :param max_stay: 最长“未改进”迭代数
        :param seed: 随机种子，方便复现实验
        """
        self.current_state = x0
        self.best_state = x0
        self.T0 = T0
        self.T = T0
        self.Tmin = Tmin
        self.max_iter = max_iter
        self.alpha = alpha
        self.max_stay = max_stay

        if seed is not None:
            random.seed(seed)

        # 这些会在 run() 中初始化
        self.current_energy: Optional[float] = None
        self.best_energy: Optional[float] = None

    # ========= 需要子类重写的核心接口 =========
    def energy(self, state: Any) -> float:
        """目标函数 / 能量函数，需要在子类中实现。"""
        raise NotImplementedError

    def neighbor(self, state: Any) -> Any:
        """产生邻域解，需要在子类中实现。"""
        raise NotImplementedError

    # ========= 可选扩展接口 =========
    def acceptance(self, E_old: float, E_new: float, T: float) -> bool:
        """
        默认的接受准则：Metropolis 准则。
        子类可以重写，比如加入玻尔兹曼因子、阈值控制等。
        """
        dE = E_new - E_old
        if dE <= 0:
            return True
        # 防止溢出
        try:
            prob = math.exp(-dE / T)
        except OverflowError:
            prob = 0.0
        return random.random() < prob

    def cool(self, T: float, k: int) -> float:
        """
        降温策略，默认几何降温：T_{k+1} = alpha * T_k。
        子类可以重写为线性降温、对数降温等。
        """
        return T * self.alpha

    def stop(self, k: int, T: float, no_improve: int) -> bool:
        """
        停止条件，可根据迭代次数、温度、未改进次数等综合判断。
        """
        if T <= self.Tmin:
            return True
        if k >= self.max_iter:
            return True
        if no_improve >= self.max_stay:
            return True
        return False

    # ========= 可选：日志 & 回调 =========
    def log_step(
        self,
        k: int,
        state: Any,
        energy: float,
        T: float,
        accepted: bool,
        is_best: bool,
    ):
        """日志接口，默认什么都不做，子类可重写写文件、打印等。"""
        pass

    # ========= 主流程 =========
    def run(
        self,
        callback: Optional[
            Callable[[int, Any, float, float, bool, bool], None]
        ] = None,
    ) -> Dict[str, Any]:
        """
        运行模拟退火。

        :param callback: 可选的回调函数，签名为
            callback(k, state, energy, T, accepted, is_best)
        :return: 包含最优解和历史信息的字典
        """
        # 初始化
        self.current_energy = self.energy(self.current_state)
        self.best_state = self.current_state
        self.best_energy = self.current_energy
        self.T = self.T0

        no_improve = 0

        history = {
            "best_energy": [],
            "current_energy": [],
            "T": [],
        }

        for k in range(1, self.max_iter + 1):
            # 1. 生成新解
            candidate_state = self.neighbor(self.current_state)
            candidate_energy = self.energy(candidate_state)

            # 2. 接受判断
            accepted = self.acceptance(self.current_energy, candidate_energy, self.T)
            if accepted:
                self.current_state = candidate_state
                self.current_energy = candidate_energy

            # 3. 更新最优解
            is_best = False
            if self.current_energy < self.best_energy:
                self.best_state = self.current_state
                self.best_energy = self.current_energy
                is_best = True
                no_improve = 0
            else:
                no_improve += 1

            # 4. 记录历史
            history["best_energy"].append(self.best_energy)
            history["current_energy"].append(self.current_energy)
            history["T"].append(self.T)

            # 5. 日志 & 回调
            self.log_step(
                k=k,
                state=self.current_state,
                energy=self.current_energy,
                T=self.T,
                accepted=accepted,
                is_best=is_best,
            )

            if callback is not None:
                callback(k, self.current_state, self.current_energy, self.T, accepted, is_best)

            # 6. 降温
            self.T = self.cool(self.T, k)

            # 7. 停止判断
            if self.stop(k, self.T, no_improve):
                break

        return {
            "best_state": self.best_state.to_json(),
            "best_energy": self.best_energy,
            "history": history,
        }
