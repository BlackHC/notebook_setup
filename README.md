# Notebook setup package

[![Build Status](https://travis-ci.org/BlackHC/mdp.svg?branch=master)](https://travis-ci.org/BlackHC/mdp)

Idea is to
```
import blackhc.notebook
```
and get someuseful stuff in your jupyter notebooks.

Right now 'useful stuff' is:

* finds the project root and changes directory to that;
* adds `$project_root/src` to the Python paths;
* load the autoreload extension and set its mode to 2.
 
## Installation

To install using pip, use:

```
pip install blackhc.notebook
```

To run the tests, use:

```
python setup.py test
```
