# Tutorials for the Natural language processing course (UL FRI)
<sup>This repository is a part of Natural Language Processing course at the University of Ljubljana, Faculty for computer and information science. Please contact [slavko.zitnik@fri.uni-lj.si](mailto:slavko.zitnik@fri.uni-lj.si) for any comments.</sub>

If you have an NVIDIA GPU, make sure NVIDIA drivers, CUDA and cuDNN are installed on your system and that versions match with PyTorch and Tensorflow..

**Anaconda installation**

Conda environment management and usage:

```
# Creation of an environment (first time only)
conda create -n nlp-course-fri python=3.6

# Activation of an environment (before running examples)
source activate nlp-course-fri

# Dependencies installation (one time only)
conda install nb_conda==2.2.1 nltk==3.6.1 matplotlib==3.3.4 bs4==4.9.3 pandas==1.1.5 mpld3==0.5.2 python-crfsuite==0.9.7 h5py==2.10.0 pydot==1.4.1 graphviz==2.40.1 gensim==3.8.3 seaborn==0.11.1 
conda install -c huggingface transformers==4.4.2
conda install -c conda-forge ipywidgets==7.6.3
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit==11.0.221 -c pytorch
pip install tqdm seqeval tensorflow==2.4.1 keras==2.4.3
conda install -c anaconda scikit-learn==0.24.1

# Separately install crfsuite if above does not work
# Download from https://pypi.org/project/sklearn-crfsuite/#files or via pip if works
# This library is not only CRFSuite wrapper but also includes CRFSuite binaries
pip install sklearn_crfsuite-0.3.6-py2.py3-none-any.whl

# Explore and run notebooks
jupyter notebook 

# Or install and run jupyter lab
conda install -c conda-forge jupyterlab
jupyter lab

# Close environment
source deactivate
```

Show existing environments:

```
conda info --envs
```

**non-Anaconda environment**

I propose that you us libraries listed above using Python 3 and `virtualenv.`