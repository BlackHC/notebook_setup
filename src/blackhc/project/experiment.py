"""
Wrapper code to make it easy to run experiments with LAAOS.

`parse_args_and_load_experiment_config` facilitates loading external experiment configs.

`embedded_experiment` returns a store for an experiment based on the current script file.

`embedded_experiments` returns a store a given job (out of number of jobs).
"""
import argparse
import functools
import os
import sys
import time
from os import path

import dataclasses

import torch
import typing
from laaos import TypeHandler

import laaos
import laaos.torch

# KEEP THIS AROUND TO SET THE DIRECTORY AUTOMAGICALLY
import blackhc.project.script


def get_likely_github_commit_url(github_url: str, commit):
    # "git@github.com:user/repo.git"
    # "https://github.com/user/repo.git"

    git_prefix = "git@github.com:"
    if github_url.startswith(git_prefix):
        github_url = f"https://github.com/{github_url[len(git_prefix)]:}"
    https_prefix = "https://github.com/"
    if not github_url.startswith(https_prefix):
        return commit

    suffix = ".git"
    if github_url.endswith(suffix):
        github_url = f"{github_url[:-len(suffix)]}"
    else:
        return commit

    return f"{github_url}/commit/{commit}"


def get_git_head_commit_and_url(path):
    try:
        import git
    except ImportError:
        return "", ""

    try:
        repo = git.Repo(path, search_parent_directories=True)
        commit = str(repo.commit())
        urls = [get_likely_github_commit_url(url, commit) for url in repo.remote("origin").urls] or [commit]
        return commit, urls[0]
    except git.exc.InvalidGitRepositoryError:
        return "", ""


def dummy_add_experiment_config_args(parser):
    return parser


def parse_args_and_load_experiment_config(
    add_experiment_config_args=dummy_add_experiment_config_args, exposed_symbols=tuple(), description="Experiment"
):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120),
    )
    parser.add_argument("--experiment_id", type=str, default=None, help="experiment id")
    parser.add_argument(
        "--experiment_config", type=str, default=None, help="Laaos file that contains all experiment task configs"
    )
    parser.add_argument(
        "--result_dir", type=str, default=None, help="Laaos file that contains all experiment task configs"
    )
    parser.add_argument(
        "--experiment_description", type=str, default="Trying stuff..", help="Description of the experiment"
    )
    parser = add_experiment_config_args(parser)
    args = parser.parse_args()

    if args.experiments_config is not None:
        config = laaos.safe_load(args.experiments_config, exposed_symbols=exposed_symbols)
        # Merge the experiment config with args.
        # Args take priority.
        args = parser.parse_args(namespace=argparse.Namespace(**config[args.experiment_id]))

    # DONT TRUNCATE LOG FILES EVER AGAIN!!! (OFC THIS HAD TO HAPPEN AND WAS PAINFUL)
    if args.experiment_id:
        store_name = f"experiment_{args.experiment_id}_result"
    else:
        store_name = f"unnamed_result"

    if args.experiments_config:
        result_dir = path.dirname(args.experiments_config) + "/results/"
    else:
        result_dir = "./experiments/runs/results"

    store = create_experiment_store(result_dir=result_dir, store_name=store_name)
    store["args"] = args.__dict__
    store["experiments_config"] = args.experiments_config
    store["experiment_id"] = args.experiment_id

    print("Parsed args:")
    print(args.__dict__)

    return args, store


def embedded_experiment(script_file):
    script_file = path.abspath(script_file)
    result_dir = path.dirname(script_file) + "/results/"
    result_name = path.splitext(path.basename(script_file))[0] + "_result"
    store = create_experiment_store(result_dir=result_dir, store_name=result_name)
    store["experiment"] = script_file
    return store


def embedded_experiments(script_file, num_jobs):
    parser = argparse.ArgumentParser(
        description="Unpacking Information Bottlenecks Experiment",
        formatter_class=functools.partial(argparse.ArgumentDefaultsHelpFormatter, width=120),
    )
    parser.add_argument("--id", type=int, default=None, help="experiment id")
    parser.add_argument(
        "--num_workers", type=int, default=None, help="number of worker (None means one worker per job)"
    )
    args = parser.parse_args()

    num_workers = args.num_workers or num_jobs
    worker_id = args.id
    job_id = args.id
    if job_id is None or not (0 <= job_id < num_jobs):
        raise ValueError(f"0 <= --id={job_id} < {num_jobs}!")

    # DONT TRUNCATE LOG FILES EVER AGAIN!!! (OFC THIS HAD TO HAPPEN AND BE PAINFUL)
    script_file = path.abspath(script_file)
    result_dir = path.dirname(script_file) + "/results/"

    while job_id < num_jobs:
        result_name = path.splitext(path.basename(script_file))[0] + f"_job_{job_id}"
        store = create_experiment_store(result_dir=result_dir, store_name=result_name)

        store["experiment"] = script_file
        store["job_id"] = job_id
        store["worker_id"] = worker_id
        store["num_workers"] = num_workers

        print(f"Job id: {job_id}/{num_jobs}")
        print(f"Worker id: {worker_id}/{num_workers}")

        yield job_id, store

        store.close()

        job_id += num_workers


def create_experiment_store(*, result_dir, store_name=None):
    # Make sure we have a directory to store the results in, and we don't crash!
    os.makedirs(result_dir, exist_ok=True)
    safe_store_name = store_name if store_name else "results"
    store = laaos.open_file_store(
        safe_store_name,
        prefix=result_dir,
        truncate=False,
        type_handlers=laaos.torch.TypeHandlers + laaos.DefaultTypeHandlers,
    )

    store["timestamp"] = int(time.time())
    store["cmdline"] = sys.argv[:]
    commit, github_url = get_git_head_commit_and_url(".")
    store["commit"] = commit
    store["github_url"] = github_url
    print("Command line:")
    print("|".join(sys.argv))
    print(f"GitHub URL: {github_url}")
    print(f"Commit: {commit}")
    print(f"Results stored in {store.uri}")
    return store
