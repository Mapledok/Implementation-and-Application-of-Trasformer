"""
@Time : 2023/4/7 14:06
@Author : 十三
@Email : mapledok@outlook.com
@File : basic.py
@Project : Transformer
"""
import torch
import warnings

warnings.filterwarnings('ignore')
RUN_EXAMPLES = True


# Define some convenience helper functions
def is_interactive_notebook():
    return __name__ == '__main__'


def show_example(fn, args=[]):
    if RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{'lr': 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False) -> None:
        None


class DummyScheduler:
    def step(self):
        None
