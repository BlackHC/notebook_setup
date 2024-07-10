from unittest.mock import patch

import pytest

import wandb
from blackhc.project.utils.log_path import xpath
from blackhc.project.utils.wandb_log_path import (
    commit,
    define_metric,
    log,
    log_metric,
    wandb_step,
)


@pytest.fixture(autouse=True)
def reset_xpath_state():
    # Reset the state of xpath to its initial state
    # This could be resetting variables or re-instantiating the object
    xpath.test_reset()

    yield  # This allows the test to run. After the test, any cleanup can occur if needed.


@pytest.mark.integration
def test_integration_wandb_log_path():
    run = wandb.init(project="dummy_project", mode="online")

    # Test wandb_step
    with wandb_step("test_step"):
        assert xpath.current_step_name == "test_step"
        assert xpath.current_step_index == 0

        # Test log_metric
        log_metric("test_metric", 1)

        # Test log
        log({"test_metric2": 2})

        # Test commit
        commit()
        # Check if commit was successful

        # Test define_metric
        define_metric("test_metric3")

    run.finish()


def test_wandb_log_path():
    with (
        patch("wandb.log"),
        patch("wandb.define_metric"),
        patch("wandb.finish"),
        patch("wandb.init"),
    ):
        wandb.init(project="dummy_project", mode="offline")

        assert xpath.path_separator == "/"

        # Test wandb_step
        with wandb_step("test_step"):
            wandb.define_metric.assert_called_with("test_step/*", "test_step/step")

            assert xpath.current_step_name == "test_step"
            assert xpath.current_step_index == 0
            wandb.log.assert_called_with({"test_step/step": 0}, commit=False)

            # Test log_metric
            log_metric("test_metric", 1)
            wandb.log.assert_called_with({"test_step/test_metric": 1}, commit=False)

            # Test log
            log({"test_metric2": 2})
            wandb.log.assert_called_with({"test_step/test_metric2": 2}, commit=False)

            # Test commit
            commit()
            # Check if commit was successful
            wandb.log.assert_called_with({}, commit=True)

            # Test define_metric
            define_metric("test_metric3")
            wandb.define_metric.assert_called_with("test_step/test_metric3")
            # Verify that except for the two calls there were no others
            assert wandb.define_metric.call_count == 2

        wandb.log.assert_called_with({}, commit=True)

        wandb.finish()
