# Tutorials for the Natural language processing course (UL FRI)
<sup>This repository is a part of Natural Language Processing course at the University of Ljubljana, Faculty for computer and information science. Please contact [slavko.zitnik@fri.uni-lj.si](mailto:slavko.zitnik@fri.uni-lj.si) for any comments.</sub>

If you have an NVIDIA GPU, make sure NVIDIA drivers, CUDA and cuDNN are installed on your system and that versions match with PyTorch and Tensorflow..

** NOTE: Libraries are being updated in the year 2021/22. I am using M1 (see below) - during the course I will also try to update for other architectures. **

**Anaconda installation (OPTION A)**

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
pip install tqdm seqeval tensorflow==2.4.1 keras==2.4.3 classla==1.1.0
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

**Anaconda installation (OPTION B)**

Create a new conda environment based on the provided `environment.yml` file:

```bash
# Creation of an environment (first time only)
conda env create -f environment.yml

# Activate environment
conda activate nlp-course-fri
```


The environment was successfully used within the following system: Ubuntu 20.04, CUDA 11.3, cuDNN 8.1.1.33-1+cuda11.2.

**non-Anaconda environment**

I propose that you us libraries listed above using Python 3 and `virtualenv.`

** Apple M1 processor: **

```
# Following installation from:
# https://developer.apple.com/metal/tensorflow-plugin/
# Download conda from https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate

# Execute if upgrading:
conda install -c apple tensorflow-deps
# uninstall existing tensorflow-macos and tensorflow-metal
python -m pip uninstall tensorflow-macos
python -m pip uninstall tensorflow-metal

# Creation of an environment (first time only)
conda create -n nlp-course-fri python=3.8

# Activation of an environment (before running examples)
source activate nlp-course-fri

conda install -c apple tensorflow-deps --force-reinstall
conda install -c apple tensorflow-deps==2.6.0
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
# TODO: Tensorflow Version updated, test needed

# Dependencies installation (one time only)
conda install nb_conda==2.2.1 
conda install nltk==3.6.1
conda install matplotlib==3.3.4
conda install bs4==4.9.3
conda install pandas==1.1.5
conda install mpld3==0.5.2
conda install python-crfsuite==0.9.7 #TODO: unsupported for python 3.8 (use above)
conda install h5py==3.1.0 #TODO: Version updated, test needed
conda install pydot==1.4.2 # TODO: Version updated, test needed
conda install graphviz==3.0.0 # TODO: Version updated, test needed
conda install gensim==3.8.3 #TODO: unsupported for python 3.8 (use above)
conda install seaborn==0.11.1
conda install -c huggingface transformers==4.11.3 # TODO: Version updated, test needed
conda install -c conda-forge ipywidgets==7.6.3
conda install pytorch=1.11.0 torchvision=0.2.2 -c pytorch # TODO: Version updated (+torchaudio not available, +CPU only), test needed
pip install tqdm seqeval classla==1.1.0 # TODO: Version updated, test needed
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