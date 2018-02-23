import sys
import os

_project_dir = None


def set_project_dir(project_dir):
    global _project_dir
    if _project_dir:
        old_src_path = get_src_path(_project_dir)
        if old_src_path in sys.path:
            sys.path.remove(old_src_path)
    _project_dir = project_dir
    src_path = get_src_path(_project_dir)
    if src_path and src_path not in sys.path:
        sys.path.append(src_path)
        print('Appended %s to paths' % src_path)

    if _project_dir:
        os.chdir(project_dir)
        print('Switched to directory %s' % project_dir)


def get_src_path(project_dir):
    return os.path.join(project_dir, 'src') if project_dir else None


# Inspired by https://stackoverflow.com/questions/19687394/python-script-to-determine-if-a-directory-is-a-git-repository
def get_git_working_dir(path):
    try:
        import git
    except ImportError:
        return None

    try:
        return git.Repo(path, search_parent_directories=True).working_tree_dir
    except git.exc.InvalidGitRepositoryError:
        return None


def get_cookiecutter_project_path(seed_path):
    # Check if we are within a notebooks sub path
    path = seed_path
    while not os.path.ismount(path):
        if os.path.basename(path) == 'notebooks':
            project_dir = os.path.dirname(path)
            return project_dir
        path = os.path.dirname(path)

    # Check if we are at the root. And there is a src subdirectory.
    if os.path.isdir(os.path.join(seed_path, 'src')):
        return os.path.abspath(seed_path)
    return None


def infer_and_set_project_dir():
    set_project_dir(get_git_working_dir(os.getcwd()) or get_cookiecutter_project_path(os.getcwd()))


infer_and_set_project_dir()
