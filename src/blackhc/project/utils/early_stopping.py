"""
MIT License

Copyright (c) 2023 Andreas 'blackhc' Kirsch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import operator
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Early stopping generator.

    We can zip this generator with the training loop to stop training early.

    :param patience: how many epochs/upddates to wait for improvement.
    :param lower_is_better: whether a lower value is better.
    """

    patience: int
    lower_is_better: bool = True

    def __post_init__(self):
        self.max_patience = self.patience
        if self.lower_is_better:
            self.best_score = float("inf")
            self.cmp = operator.lt
        else:
            self.best_score = -float("inf")
            self.cmp = operator.gt

    @property
    def patience_remaining(self) -> bool:
        return self.patience >= 0

    def step(self, score):
        assert self.patience_remaining
        if self.cmp(score, self.best_score):
            self.best_score = score
            self.patience = self.max_patience
        else:
            self.patience -= 1

        return self.patience_remaining

    def __iter__(self):
        step_counter = 0
        while self.patience_remaining:
            yield self.step
            step_counter += 1
        print(
            f"Early stopping with best score {self.best_score} at step {step_counter}"
        )
