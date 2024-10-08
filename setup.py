# Always prefer setuptools over distutils
from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="blackhc.project",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="8.0.0",
    description="Notebook setup code",
    # Fix windows newlines.
    long_description=long_description.replace("\r\n", "\n"),
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://github.com/blackhc/notebook_setup",
    # Author details
    author="Andreas @blackhc Kirsch",
    author_email="blackhc+notebook_setup@gmail.com",
    # Choose your license
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    # What does your project relate to?
    keywords="jupyter",
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=[
        "blackhc.project",
        "blackhc.project.utils",
        "blackhc.project.utils.tests",
    ],
    package_dir={"": "src"},
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "gitpython",
        "fs.sshfs",
        "laaos",
        "jsonpickle",
        "toolz",  # Functional programming
        # See https://x.com/BlackHC/status/1780995421852127477.
        "jupyter_client<8.0.0",  # This fixes issues with jupyter_client <-> pyzmq
        "pyzmq<25",  # This fixes issues with jupyter_client <-> pyzmq
        "objproxies",
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e ".[dev,test]"
    extras_require={
        "dev": ["check-manifest"],
        "test": [
            "coverage",
            "pytest",
            "pytest-forked",
            "pyfakefs",
            "torch",
            "psutil",
            "pytorch-ignite",
            "blackhc.progress_bar",
            "wandb",
            "prettyprinter",
            "rich",
            "markdownify",
            "ipython",
        ],
    },
    setup_requires=[
        "pytest-runner",
    ],
)
