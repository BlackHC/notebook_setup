"""
Logging execution paths to make it easy to create nested paths for logging and debugging.

This module provides a `LogPath` class (aliased as `xpath`) that allows for hierarchical
logging of execution paths. It's particularly useful for creating nested paths for
logging and debugging in complex applications.

Key features:
- Create nested execution steps with unique identifiers
- Track summary steps separately from regular steps
- Automatically measure and record execution time for each step
- Generate unique metric names based on the current execution path
- Maintain statistics for step execution times

Usage:
```
    from blackhc.project.utils.log_path import xpath

    with xpath.step("main_process"):
        # Do some work
        with xpath.step("sub_process"):
            # Do sub-process work
            metric_name = xpath.metric_name("accuracy")
            # Log metric_name with your logging system

    # Access statistics
    print(xpath.all_step_names)
    print(xpath.all_step_stats)
```

Classes:
    LogPathStep: Represents a single step in the execution path
    LogPathNodeStats: Stores statistics for a specific step
    LogPath: Main class for managing the execution path and logging

Note: This module uses ContextVar for thread-safe context management.
"""
import dataclasses
import warnings
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from blackhc.project.utils.context_stopwatch import ContextStopwatch


@dataclass
class LogPathStep:
    step_name: str
    step_index: int
    unique_path: str
    is_summary_step: bool = False
    parent: "LogPathStep | None" = None
    used_names: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )

    def create_child(
        self, name, *, path_separator: str, is_summary_step: bool = False
    ) -> "LogPathStep":
        is_summary_step |= self.is_summary_step

        step_index = self.used_names[name]

        # Built a unique name that is fully indexed.
        if is_summary_step:
            unique_name = name

            if self.used_names[name] >= 1:
                unique_name = f"{name}+{step_index}"

                if step_index == 1:
                    # Log a warning that the name has been used before.
                    warnings.warn(
                        f"Sub name {name} has already been used in summary step. Use a unique name please!"
                    )
        else:
            unique_name = f"{name}[{step_index}]"

        # Increment the number of times the sub_name has been used.
        self.used_names[name] += 1

        if self.parent is None:
            step_name = name
            unique_path = unique_name
        else:
            step_name = f"{self.step_name}{path_separator}{name}"
            unique_path = f"{self.unique_path}{path_separator}{unique_name}"

        if is_summary_step:
            step_name = unique_path

        return LogPathStep(step_name, step_index, unique_path, is_summary_step, self)

    def metric_name(
        self, metric_name, *, path_separator: str, is_summary: bool = False
    ):
        if self.parent is None:
            return metric_name
        else:
            if is_summary:
                return f"{self.unique_path}{path_separator}{metric_name}"
            else:
                return f"{self.step_name}{path_separator}{metric_name}"


@dataclass
class LogPathNodeStats:
    count: int = 0
    first_time: float = float("nan")
    rest_time: float = float("nan")

    @property
    def mean_rest_time(self):
        if self.count == 0:
            return self.rest_time
        return self.rest_time / (self.count - 1)

    @property
    def mean_time(self):
        if self.count <= 1:
            return self.first_time
        return (self.first_time + self.rest_time) / self.count

    @property
    def total_time(self):
        if self.count <= 1:
            return self.first_time
        return self.first_time + self.rest_time

    def append_time(self, time):
        if self.count == 0:
            self.first_time = time
        else:
            if self.count == 1:
                self.rest_time = 0.0
            self.rest_time += time

        self.count += 1


class LogPath:
    """
    Log Execution Paths Manager.

    A simple class to make it easy to create nested paths for logging and debugging.
    """

    path_separator = "/"
    _ROOT: LogPathStep = LogPathStep("", 0, "")
    _current: ContextVar[LogPathStep] = ContextVar("current_xpath_node", default=_ROOT)
    all_summary_step_names = set()
    all_step_names = set()
    all_metrics = set()
    all_summary_metrics = set()
    all_step_stats = defaultdict(LogPathNodeStats)

    @classmethod
    @contextmanager
    def step(cls, name, is_summary_step: bool = False):
        """
        Create a new sub-path.

        Args:
            name: The name of the step.
            is_summary_step: If True, the step will be treated as a summary step.
                This means that a unique name will be generated for the step.
        """
        current = cls._current.get()
        child = current.create_child(
            name, path_separator=cls.path_separator, is_summary_step=is_summary_step
        )
        if child.is_summary_step:
            cls.all_summary_step_names |= {child.step_name}
        else:
            cls.all_step_names |= {child.step_name}
        token = cls._current.set(child)
        try:
            with ContextStopwatch() as stopwatch:
                yield child
            cls.all_step_stats[child.step_name].append_time(stopwatch.elapsed_time)
        finally:
            cls._current.reset(token)

    @classmethod
    @property
    def current_step_name(cls):
        """
        Get the current path.

        Returns:
            The current path.
        """
        return cls._current.get().step_name

    @classmethod
    @property
    def current_step_index(cls):
        """
        Get the current step index.

        Returns:
            The current step index.
        """
        return cls._current.get().step_index

    @classmethod
    @property
    def current_step_count(cls):
        """
        Get the current step count.

        Returns:
            The current step count.
        """
        return cls.all_step_stats[cls.current_step_name].count

    @classmethod
    @property
    def current_unique_path(cls):
        """
        Get the current unique path.

        Returns:
            The current unique path.
        """
        return cls._current.get().unique_path

    @classmethod
    def metric_name(cls, metric_name, is_summary: bool = False):
        """
        Get the full metric name.

        Args:
            metric_name:

        Returns:
            The full metric name.
        """
        metric_name = cls._current.get().metric_name(
            metric_name, path_separator=cls.path_separator, is_summary=is_summary
        )
        if is_summary:
            cls.all_summary_metrics |= {metric_name}
        else:
            cls.all_metrics |= {metric_name}
        return metric_name

    @classmethod
    def test_reset(cls):
        cls._current.set(cls._ROOT)
        cls._ROOT.used_names.clear()
        cls.all_step_names = set()
        cls.all_summary_step_names = set()
        cls.all_step_stats.clear()
        cls.all_metrics = set()
        cls.all_summary_metrics = set()


xpath = LogPath

# Limit exports
__all__ = ["xpath"]
