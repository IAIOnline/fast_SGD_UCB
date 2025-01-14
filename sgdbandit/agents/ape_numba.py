"""
APE algorithm from https://arxiv.org/pdf/2010.12866

!!! Bagged version, use simple APE
"""

from typing import Any
import numpy as np
from numba import njit, float64
from numba.typed import List
from numba.experimental import jitclass

import scipy.stats as sps # for perturbation

from .abstract_agent import AbstractAgent

spec = [('p', float64), ("b", float64)]
@jitclass(spec)
class psi_p:
    def __init__(self, p) -> None:
        self.p = p
        if p < 1 + 1e-6:
            self.b: float64 = 1.
        else:
            self.b: float64 = ( 2 * ((2-p)/(p-1))**(1 - 2/p) + ((2-p)/(p-1))**(2 - 2/p) )**(-p/2)
    
    def calc(self, x: np.array):
        """
        there x may be a vector of values.
        in this case we get sum of them before return
        """
        # if  not isinstance(x, np.ndarray):
            # x = np.array([x])
        positives = (x >= 0)
        psi = np.sum(np.log(self.b * np.power(np.abs(x[positives]), self.p) + x[positives] + 1)) +\
            np.sum(np.log(self.b * np.power(np.abs(x[~positives]), self.p) - x[~positives] + 1))
        
        return psi

    # def __call__(self, x) -> Any:
    #     return calc(x)        


class APENumba(AbstractAgent):
    def __init__(self, n_actions, c, p, F_inv = None):
        super().__init__(n_actions,)
        if F_inv is None:
            F_inv = lambda x: sps.chi2.ppf(x, df = 1, scale = 2)
        self.F_inv = F_inv # for perturbation generation
        self.c = c
        self.p = p
        self.psi = psi_p(p)
        self._initial_exploration = np.random.permutation(n_actions)

        self._rewards = List([List.empty_list(float64) for i in range(self.n_actions)])
    def reset(self):
        self.__init__(self.n_actions, self.c, self.p, self.F_inv)

    @property
    def get_perturbation(self):
        u = np.random.rand(self.n_actions)
        return self.F_inv(u)

    @staticmethod  
    @njit
    def inner_p_robust(arm_est, history_pull, rewards, c, p, psi):
        for i, (n_i, r_i) in enumerate(zip(history_pull, rewards)):
            coeff = (c * (n_i**(1/p)))
            rewards_i = List([r / coeff for r in r_i])

            arm_est[i] += c / (n_i ** (1 - 1/p)) * \
                psi.calc(rewards_i)
        return np.argmax(arm_est)

    def p_robust_estimation(self):
        """
        here we cmpute estimation of reward for all arms        
        """
        arm_est = self.get_perturbation * \
                    self.c / (self._history_pull ** (1 - 1/self.p))
        robust_est_res = self.inner_p_robust(arm_est, self._history_pull, self._rewards, self.c, self.p, self.psi)

        return robust_est_res


    def get_action(self):
        # Место для Вашей реализации
        t = self._total_pulls
        if t < self.n_actions:
            return self._initial_exploration[t]
        else:
            return self.p_robust_estimation()
    