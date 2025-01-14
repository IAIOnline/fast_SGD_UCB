# Clipped-Med-SMD method from
# https://arxiv.org/pdf/2402.02461

from copy import deepcopy
import math
from collections import defaultdict

from scipy import optimize
import numpy as np
from ..utils.arms import BaseArm
from .abstract_agent import AbstractAgent

def tsallis_prox_12(loss_vec, normalizer_const = 0.,):
        def normalizer(x):
            # self.eta_t  = 1
            # print(np.sum(loss_vec - x) ** 0.5)
            return ((np.sum((loss_vec - x) ** (-2))) - 1) ** 2

        normalizer_const = optimize.newton(normalizer, x0= normalizer_const)

        probs = (loss_vec - normalizer_const)**(-2)

        # for i in range(len(probs)):
            # probs[i] = (loss_vec[i] - normalizer_const)**(-2)
        probs = probs/np.sum(probs)
        # assert  abs(sum(probs) - 1.) < 1e-6
        # print(loss_vec, probs)
        return probs, normalizer_const
        
def _clip(vect, lambd):
    norm = np.linalg.norm(
        vect, float("inf")
    )
    if norm == 0:
        return np.zeros_like(vect, dtype=float)
    else:
        return min(1, lambd / norm) * vect

class ClippedMedSmd(AbstractAgent):
    def __init__(self, n_actions: int, 
                T: int,
                median_count: int,
                stepsize: float, 
                clip_lambda: float,
                prox_function = tsallis_prox_12) -> None:
        """

        Parameters:
        :parameter n_actions: number of actions
        :parameter T: number of  steps
        :parameter median_count: 2m+1 is a number of pulls to compute median
        :parameter prox_function: callable, that computes updated point with 
                old point and gradient estimation. __call__(old_point, grad_estimation)

        """
        super().__init__(n_actions, False)
        self._total_calls = 0  # this is the number of times the arms has been selected
        self.n_actions = n_actions
        self.T = T
        self.median_count = median_count
        self.prox_func = prox_function
        self.stepsize = stepsize
        self.clip_lambda = clip_lambda

        self.alpha = 1/(self.T)

        self.arm_probs = np.ones(self.n_actions, float)/self.n_actions
        self.normalizer_const = 0.
        self._unselect()

    def reset(self):
        self.arm_probs = np.ones(self.n_actions, float)/self.n_actions
        self.normalizer_const = 0.
        self._unselect()

    def _unselect(self):
        self.arm_pulls_grad = defaultdict(list)
        self.last_pulls = 0

    def get_action(self):
            
            # print("lol")
        arm_t = np.random.choice(self.n_actions, 1, p=list(self.arm_probs))[0]
        self.last_pulls += 1
        return arm_t

    def _update(self):
        arm_grads = np.zeros_like(self.arm_probs)
        for arm in range(self.n_actions):
            arm_rews = self.arm_pulls_grad[arm]
            val = np.median(arm_rews) if len(arm_rews) > 0 else 0.
            arm_grads[arm] = val
        
        arm_grads = _clip(arm_grads, self.clip_lambda)
        
        loss_vec =  1./self.arm_probs**0.5 - self.stepsize * arm_grads
        self.arm_probs, self.normalizer_const = tsallis_prox_12(loss_vec, self.normalizer_const)
        # self.arm_probs = (1 - self.alpha) * self.arm_probs + self.alpha/self.n_actions

    def update(self, action, reward):
        super().update(action, reward)
        self.arm_pulls_grad[action].append(reward / self.arm_probs[action])

        if self.last_pulls >= self.median_count * 2 + 1:
            self._update()
            self._unselect()
        self._total_calls += 1

        # if self.selected:
        #     self.selected_rewards.append(reward)
        #     self.selected_count += 1
        #     if self._total_calls < self.n_actions and self._init_steps:
        #         if self.selected_count >= self._init_steps:
        #             self._init_update()
        #             self._un_select()
        #             self._total_calls += 1
        #     else:
        #         if self.selected_count >= self._clear_arm.pulls_for_update:
        #             self._update()
        #             self._un_select()
        #             self._total_calls += 1
