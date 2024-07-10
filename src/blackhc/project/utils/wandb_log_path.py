"""
Weights & Biases (wandb) Integration for Logging with XPath

This module provides utility functions and context managers to integrate
Weights & Biases (wandb) logging with a custom XPath logging system. It
offers convenient wrappers around wandb functions to log metrics, define
custom steps, and manage the logging hierarchy.

Key features:
- Custom step management with `wandb_step`
- Metric logging with automatic path handling
- Support for summary and non-summary metrics
- Integration with an XPath-based logging system

Dependencies:
- wandb
- blackhc.project.utils.log_path (xpath)

Note: This module requires the 'wandb' package to be installed.
"""
import functools
from contextlib import contextmanager

try:
    import wandb
except ImportError:
    raise ImportError("Package 'wandb' is not installed. Please install it using 'pip install wandb'")

from blackhc.project.utils.log_path import xpath


@contextmanager
def wandb_step(name):
    """
    A context manager for wandb steps.


    Args:
        name: The name of the step.
        is_summary_step: Whether this is a summary step.

    Returns:
        The context manager.
    """

    with xpath.step(name, is_summary_step=False):
        step_metric = xpath.metric_name("step", is_summary=False)
        wandb.log({step_metric: xpath.current_step_index}, commit=False)
        if xpath.current_step_count == 0:
            wandb.define_metric(f"{xpath.current_step_name}/*", step_metric)
        try:
            yield
        finally:
            commit()


def log_metric(metric_name, value, is_summary=False):
    """
    Log a metric to wandb.

    Args:
        metric_name: The metric name.
        value: The metric value.
        is_summary: Whether this is a summary metric.
    """
    metric_name = xpath.metric_name(metric_name, is_summary=is_summary)
    if is_summary is False:
        wandb.log({metric_name: value}, commit=False)
    else:
        wandb.run.summary[metric_name] = value


def log(metrics, is_summary=False):
    """
    Log metrics to wandb.

    Args:
        metrics: The metrics.
        is_summary: Whether this is a summary metric.
    """
    metrics = {
        xpath.metric_name(k, is_summary=is_summary): v for k, v in metrics.items()
    }
    if is_summary is False:
        return wandb.log(metrics, commit=False)
    else:
        return wandb.run.summary.update(metrics)


def commit():
    """
    Commit the metrics to wandb.
    """
    return wandb.log({}, commit=True)


@functools.wraps(wandb.define_metric)
def define_metric(metric_name, *args, **kwargs):
    """
    Define a metric for wandb.

    Args:
        metric_name: The metric name.
        ...: The other arguments.
    """
    metric_name = xpath.metric_name(metric_name)
    return wandb.define_metric(metric_name, *args, **kwargs)
