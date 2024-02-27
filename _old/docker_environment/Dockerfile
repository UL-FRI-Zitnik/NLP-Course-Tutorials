FROM python:3.7-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8

WORKDIR /app

# Copy files and finish install
ADD ./install-models-and-data.py .
ADD ./sklearn_crfsuite-0.3.6-py2.py3-none-any.whl .

RUN pip install jupyter==1.0.0 tqdm==4.65.0 seqeval==1.2.2 keras==2.4.3 classla==1.1.0 \
    scikit-learn==0.24.1 nltk==3.6.1 matplotlib==3.3.4 beautifulsoup4==4.9.3 \
    pandas==1.1.5 mpld3==0.5.2 h5py==2.10.0 pydot==1.4.1 gensim==3.8.3 seaborn==0.11.1 \
    graphviz==0.20.1 ipywidgets==7.6.3 transformers==4.4.2 tensorflow==2.4.1 \
    torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 sentencepiece==0.1.97 \
    datasets==2.11.0 rouge-score==0.1.2 openai==0.27.4 \
    && pip install sklearn_crfsuite-0.3.6-py2.py3-none-any.whl \
    && python install-models-and-data.py \
    && pip uninstall tokenizers -y

EXPOSE 8888

ENTRYPOINT jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser \
            --NotebookApp.token=${JUPYTER_PASSWORD} --allow-root --notebook-dir=${JUPYTER_DIR}