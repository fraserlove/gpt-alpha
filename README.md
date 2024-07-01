# GPT
A full implementation of a Generative Pre-trained Transformer (GPT) model in PyTorch. The model is trained on the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 10B token dataset.

## Installation and Usage
Run the following to install the GPT and its required dependencies:
```bash
git clone https://github.com/fraserlove/gpt.git
cd gpt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Tokenisation
The `tiktoken` library is used as the default tokeniser. Alternatively, a custom tokeniser is available within `gpt.tokeniser`. To train the custom tokeniser on the `fineweb-edu` dataset, run the following command:
```bash
python gpt/tokeniser.py
```

### Dataset
The `fineweb-edu` dataset is used as the dataset for training the GPT model as it is a large-scale, high quality dataset of educational content. By default, the `sample-10B` version of the dataset is used, which contains 10B tokens. The dataset is available on the Hugging Face Datasets Hub and can be downloaded using the `datasets` library. The dataset is stored in shards in the `cache/fineweb_edu_10B` directory. To download the dataset, run the following command:
```bash
python fineweb.py
```

### Training
The GPT model can be trained using the `train.py` script. The script supports both single-GPU and multi-GPU training using data parallelism. The model is trained using a custom training loop with a learning rate scheduler and gradient clipping as per GPT-3.

To run on a single GPU, use the following command:
```bash
python train.py
```

To run on multiple GPUs, use the following command:
```bash
torchrun --standalone --nproc_per_node={n_gpus} train.py
```
where `n_gpus` is the number of GPUs to use in training.