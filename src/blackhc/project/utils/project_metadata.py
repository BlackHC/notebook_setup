"""
Project metadata collector.

Collects different metadata (from git and wandb) to make it easy to pin down an experiment.
"""
from dataclasses import dataclass
from blackhc.project import project_dir


@dataclass
class ProjectMetadata:
    project_dir: str | None
    git_commit: str | None
    git_url: str | None
    wandb_project: str | None
    wandb_url: str | None