from blackhc.project import infer_and_set_project_dir, is_run_from_ipython, setup_autoreload

infer_and_set_project_dir()

if is_run_from_ipython():
    # Only execute auto_reload if we are in ipython
    setup_autoreload()