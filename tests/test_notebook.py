from pyfakefs import fake_filesystem
from blackhc import notebook
import os


def test_get_cookiecutter_project_path_from_notebooks(fs: fake_filesystem.FakeFilesystem):
    fs.CreateDirectory('/tmp/blackhc.notebook/notebooks')
    assert (notebook.get_cookiecutter_project_path('/tmp/blackhc.notebook/notebooks') == os.path.abspath(
        '/tmp/blackhc.notebook'))


def test_get_cookiecutter_project_path_with_src(fs: fake_filesystem.FakeFilesystem):
    fs.CreateDirectory('/tmp/blackhc.notebook/src')
    assert (
        notebook.get_cookiecutter_project_path('/tmp/blackhc.notebook/') == os.path.abspath('/tmp/blackhc.notebook'))


def test_get_cookiecutter_project_path_without_src(fs: fake_filesystem.FakeFilesystem):
    fs.CreateDirectory('/tmp/blackhc.notebook')
    assert notebook.get_cookiecutter_project_path('/tmp/blackhc.notebook/') is None
