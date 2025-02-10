# pl-llm
Protein ligand LLM

More details about architecture: https://docs.google.com/presentation/d/14BxXoiM0FnhG8YOCI9j-X6y_6BjM8FfPXUGFslo6lhY/edit?usp=sharing

Perform training using prot_ligand.py:
```
Script that trains model on protein-ligand interactions represented via ic50. Make sure you have wandb on
your machine and you are logged in.

options:
  -h, --help            show this help message and exit
  --wandb_name WANDB_NAME
                        Name of run on wandb platform
  --ds_path DS_PATH     Path of the dataset in huggingface datasets format
```
