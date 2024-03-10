from pyfakefs import fake_filesystem
from blackhc import project
import os


def test_get_cookiecutter_project_path_from_notebooks(fs: fake_filesystem.FakeFilesystem):
    fs.CreateDirectory('/tmp/blackhc.project/notebooks')
    assert (project.get_cookiecutter_project_path('/tmp/blackhc.project/notebooks') == os.path.abspath(
        '/tmp/blackhc.project'))


def test_get_cookiecutter_project_path_from_scripts(fs: fake_filesystem.FakeFilesystem):
    fs.CreateDirectory('/tmp/blackhc.project/scripts')
    assert (project.get_cookiecutter_project_path('/tmp/blackhc.project/notebooks') == os.path.abspath(
        '/tmp/blackhc.project'))


def test_get_cookiecutter_project_path_with_src(fs: fake_filesystem.FakeFilesystem):
    fs.CreateDirectory('/tmp/blackhc.project/src')
    assert (
            project.get_cookiecutter_project_path('/tmp/blackhc.project/') == os.path.abspath('/tmp/blackhc.project'))


def test_get_cookiecutter_project_path_without_src(fs: fake_filesystem.FakeFilesystem):
    fs.CreateDirectory('/tmp/blackhc.project')
    assert project.get_cookiecutter_project_path('/tmp/blackhc.project/') is None
