# Copyright (c) 2025 Harbin Institute of Technology, Huawei Noah's Ark Lab
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

from arguments import OptimizationParams


class TrainingScheduler:
    """Schedule Gaussian pruning during training."""

    def __init__(self, opt: OptimizationParams) -> None:
        self.pruning_target = None
        self.pruning_budget = None
        self.final_pruning_iteration = opt.pruning_stop_iter

    def calculate_pruning_target(self, cur_n_gaussian, target):
        self.pruning_target = target if target is not None else 0.1 * cur_n_gaussian
        self.pruning_budget = cur_n_gaussian - self.pruning_target

    def get_pruning_rate(self, iteration, cur_n_gaussian, n, uniform=False):
        if n >= 5:
            next_n_gaussian = self.pruning_target
            return max((cur_n_gaussian - next_n_gaussian) / cur_n_gaussian, 0.0)
        elif iteration >= self.final_pruning_iteration:
            return 0
        elif uniform:
            next_n_gaussian = cur_n_gaussian - int(self.pruning_budget / 5)
        else:
            next_n_gaussian = cur_n_gaussian - int(self.pruning_budget / 2**n)

        pruning_rate = max((cur_n_gaussian - next_n_gaussian) / cur_n_gaussian, 0.0)
        print("next_n_gaussian:{}".format(next_n_gaussian))
        print("ratio:{}".format(pruning_rate))
        return pruning_rate
