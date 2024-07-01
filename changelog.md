# Modifications to the original open-source repository

- **1/7/2024**
    - engine_pretrained.py: Added code regarding entropy fine-tuning
    - find_average_size.py: Added code for computing the average size of a compressed embedding
    - main_compress.py: Added code to launch entropy fine-tuning
    - main_finetune.py: Added code to fine-tune on the UCM and Potsdam datasets
    - models_mae.py: Added models with different compression bottlenecks
    - Semantic Segmentation/: Various configurations and code for fine-tuning from a compressed model (quantized or entropy fine-tuned)