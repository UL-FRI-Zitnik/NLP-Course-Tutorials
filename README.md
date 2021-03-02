# Tutorials for the Natural language processing course (UL FRI)
<sup>This repository is a part of Natural Language Processing course at the University of Ljubljana, Faculty for computer and information science. Please contact [slavko.zitnik@fri.uni-lj.si](mailto:slavko.zitnik@fri.uni-lj.si) for any comments.</sub>

TODO: add description, clean the repository, sort out dependencies + nltk resources...

**Anaconda installation**

Conda environment management and usage:

```
# Creation of an environment (first time only)
conda create -n nlp-course-fri python=3.6

# Activation of an environment (before running examples)
source activate nlp-course-fri

# Dependencies installation (one time only)
conda install nb_conda nltk matplotlib bs4
conda install -c anaconda scikit-learn

# Explore and run notebooks
jupyter notebook 

# Close environment
source deactivate
```

Show existing environments:

```
conda info --envs
```

**Pure Python 3 installation**

We are using Python 3, so first check, what is the default python interpreter on your machine. Go to console and run `python` (there may be more interpreters installed on machine and Python 3.5 might be run also using `python3.5`).

You should see output similar to the following: 

```
quaternion:~ slavkoz$ python3.5
Python 3.5.2 (v3.5.2:4def2a2901a5, Mar 2 2021, 10:47:25)
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

Then I propose you to use `virtualenv` and pip to install libraries.