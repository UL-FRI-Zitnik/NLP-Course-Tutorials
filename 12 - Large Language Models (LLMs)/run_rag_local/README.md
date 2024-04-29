### Running Local Versions of RAG Notebooks on Slurm and Singularity

#### Creation of the `singularity description file` (SIF)

To create a Singularity Image File (SIF), start by defining the base environment in a Singularity definition file:

```plaintext
Bootstrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04
```

#### Package Selection and Installation

The necessary Python packages for our project can be installed as follows:

```bash
pip3 install --upgrade pip
pip3 install numpy pandas scikit-learn
pip3 install trl transformers accelerate
pip3 install git+https://github.com/huggingface/peft.git
pip3 install datasets bitsandbytes langchain sentence-transformers
pip3 install beautifulsoup4 lxml
```

#### Example of a Singularity Definition File

Here is how your complete Singularity definition file (`container_llm.def`) might look:

```plaintext
Bootstrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04

%post
    apt-get update && apt-get install -y python3 python3-pip git gcc
    apt-get clean && rm -rf /var/lib/apt/lists/*
    pip3 install --upgrade pip
    pip3 install numpy pandas scikit-learn
    pip3 install trl transformers accelerate
    pip3 install git+https://github.com/huggingface/peft.git
    pip3 install datasets bitsandbytes langchain sentence-transformers
    pip3 install beautifulsoup4 lxml

%environment
    export PATH=/usr/local/bin:$PATH

%runscript
    echo "Running script $*"
    exec python3 "$@"
```

Build the container by running:

```bash
singularity build container_llm.sif container_llm.def
```

#### Executing the Script

Once the container is built, execute it on a GPU-enabled machine with:

```bash
singularity exec --nv container_llm.sif python3 retrieval_augmented_generation.py
```

#### Managing Additional Package Installations

If additional packages are required, you can either:
1. Rebuild the container.
2. Create a persistent overlay.

To create a one gigabyte overlay:

```bash
singularity overlay create --size 1024 overlay.img
```

Shell into this overlay and install additional packages:

```bash
singularity shell --overlay overlay.img container_llm.sif
Singularity> pip3 install faiss-gpu
Singularity> exit
```

Finally, execute the script using the overlay:

```bash
singularity exec --nv --overlay overlay.img container_llm.sif python3 retrieval_augmented_generation.py
```


### Running on SLURM

We can run the command on slurm by creating slurm job description file as bellow:


```bash
#!/bin/bash
#SBATCH --job-name=RAG-job           # Job name
#SBATCH --partition=gpu              # Partition (queue) name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=1            # CPU cores/threads per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --mem=4G                     # Job memory request
#SBATCH --time=02:00:00              # Time limit hrs:min:sec
#SBATCH --output=RAG-job_%j.log      # Standard output and error log

singularity exec --nv container_llm.sif python3 retrieval_augmented_generation.py
```


and running with `sbatch slurm.bash`.