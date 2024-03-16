
# Stable diffusion prompt generation

Generate complete prompts from instructions 


## Installation

Create conda env
```bash
  conda create --name sd_prompt_generation
```
Install library
```bash
  pip install -r requirements.txt
```
    ## Running code
Train model
```bash
    config_path = 'config/test.yaml'
    !python main.py --train_only 1 --config {config_path}
```

Eval model
```bash
    config_path = 'config/test.yaml'
    !python main.py --eval_only 1 --config {config_path}
```

Inference
```bash
    config_path = 'config/test.yaml'
    !python main.py --infer_only 1 --config {config_path} --infer_data "Your instruction"
```

