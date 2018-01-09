import sys
import os

_notebook_path = None


def set_notebook_path(notebook_path):
    global _notebook_path
    if _notebook_path:
        old_src_path = get_src_path()
        if old_src_path in sys.path:
            sys.path.remove(old_src_path)
    _notebook_path = notebook_path
    src_path = get_src_path(_notebook_path)
    if src_path and src_path not in sys.path:
        sys.path.append(src_path)


def get_src_path(notebook_path):
    p = notebook_path
    while not p.ismount():
        if p.basename() == 'notebooks':
            return p.dirname().joinpath('src')
        p = p.dirname()
    return None


set_notebook_path(os.getcwd())
