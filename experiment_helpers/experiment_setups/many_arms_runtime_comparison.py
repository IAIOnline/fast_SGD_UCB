"""
In this experiment alogorithms runtime will be compared in presence
of large amount of arms.

запустим алгоритмы на 10к и 100к ручках
на большое число шагов ~ 5000
"""

import json
import sys
from pathlib import Path
from time import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from sgdbandit import agents, environments

RANDOM_SEED = 123


def main():
    action_counts = [10, 1000, 5_000, 10_000]
    n_steps = [5_000, 10_000, 20_000,]

    result = {}
    for T in tqdm(n_steps, desc=f"time"):
        result[T] = {}

        for n_actions in tqdm(action_counts, desc=f"num actions", leave = False):

            reward_arr = np.linspace(0, 10, n_actions)
            env = environments.CauchyDistributionEnv(reward_arr=reward_arr, gamma=1)
            
            result[T][n_actions] = {}

            agent_list = [
                # agents.RobustUCBMedian(n_actions=n_actions, eps=0.0, v=10),
                # agents.SGD_SMoM(n_actions, m=0, n=1, coeff=0.1, T=T, init_steps=3, R=10),
                # agents.SGD_SMoM(n_actions, m=1, n=1, coeff=0.1, T=T, init_steps=3, R=10),
                # agents.SGD_SMoM(n_actions, m=1, n=2, coeff=0.1, T=T, init_steps=3, R=10),
                agents.APE(n_actions, c = 2, p = 1 + 0.25,),
                # agents.APE(n_actions, c = 1, p = 1 + 1.,),
                ]
            agent_names = [
                # "RUCB",
                # "SGD-UCB 0.1",
                # "SGD-UCB-Median 0.1",
                # "SGD-UCB-SMoM 0.1",
                "APE +0.25",
                # "APE 2",
                ]

            assert len(agent_list)  == len(agent_names)

            # there run algorithms without parallelization.
            for agent, name in tqdm(zip(agent_list, agent_names), desc="agent", leave = False):
                time_start = time()
                for _ in tqdm(range(T), leave = False):
                    action = agent.get_action()
                    reward = env.pull(action)
                    agent.update(action, reward)
                
                runtime = time() - time_start

                del agent
                result[T][n_actions][name] = runtime
    return result


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please, provide experiment name."
    exp_name = sys.argv[1]
    assert exp_name.endswith('.json'), "Experiment should be saved as json."
    exp_name = Path(exp_name)

    if exp_name.exists():
        raise NameError(f"{exp_name} already exists.")    

    rez = main()
    with open(exp_name, "w") as f:
        json.dump(rez, f)
