# gpt


## Installation and Usage
Setup your environment and install the required dependencies as follows:

1. **Clone the Repository:**
```bash
git clone https://github.com/fraserlove/gpt.git
cd gpt
```

2. **Create a Python Virtual Environment:**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install Dependencies via PIP:**
```bash
pip install -r requirements.txt
```
Note that `tiktoken` is used as the default tokeniser. A custom tokeniser is also available within `gpt.tokeniser` and uses the `regex` package. Finally, the package `datasets` is used to load the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 10B token dataset. The custom tokeniser can be trained on the fineweb-edu dataset by running: 
```bash
python gpt/tokeniser.py`
```

4. **Download the fineweb-edu Dataset:**
```bash
python fineweb.py
```

5. **Train the Model:**
   
For single-GPU training:
```bash
python train.py
```
For multi-GPU training:
```bash
torchrun --standalone --nproc_per_node={n_gpus} train.py
```
where `n_gpus` is the number of GPUs to use in training.

