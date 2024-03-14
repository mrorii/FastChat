# Japanese Safety-Bench

## Basic usage

### Installation

```bash
git clone https://github.com/mrorii/FastChat.git
cd FastChat
git checkout safety-bench

# If you are running on Mac:
brew install rust cmake

python3 -mvenv .venv
source .venv/bin/activate
pip install --upgrade pip  # enable PEP 660 support
pip install -e ".[model_worker,webui]"
```

### Chat with the model in CLI

```bash
python -m fastchat.serve.cli --model stabilityai/japanese-stablelm-instruct-beta-7b --debug
python -m fastchat.serve.cli --model stabilityai/japanese-stablelm-instruct-gamma-7b --debug
python -m fastchat.serve.cli --model elyza/ELYZA-japanese-Llama-2-7b-instruct --debug
```

### Chat with the model in Web UI

```bash
python -m fastchat.serve.controller
python -m fastchat.serve.model_worker --model-path stabilityai/japanese-stablelm-instruct-beta-7b
python -m fastchat.serve.gradio_web_server

# Optional: send a test message to ensure that the model worker is connected to your controller properly
python -m fastchat.serve.test_message --model-name japanese-stablelm-instruct-beta-7b
```

### Run safety evaluation benchmark

```bash
# Step 1. Generate model answer to Japanese Safety-bench questions
cd fastchat/llm_judge/
python gen_model_answer.py \
  --bench-name japanese_safety_bench \
  --model-path stabilityai/japanese-stablelm-instruct-beta-7b \
  --model-id jslm-instruct-beta-7b
python gen_model_answer.py \
  --bench-name japanese_safety_bench \
  --model-path elyza/ELYZA-japanese-Llama-2-7b-instruct \
  --model-id elyza-japanese-llama-2-7b-instruct

# Step 2. Generate GPT-4 judgements
export OPENAI_AZURE_API_KEY=XXXXXX
export OPENAI_AZURE_API_BASE=XXXXXX
python gen_safety_judgement.py --model-list [LIST-OF-MODEL-ID]

# e.g.
python gen_safety_judgement.py --model-list elyza-japanese-llama-2-7b-instruct

# Step 3. Show scores
TODO
```

### Create instruction tuning training data

```bash
cd fastchat/llm_judge/

export OPENAI_AZURE_API_KEY=XXXXXX
export OPENAI_AZURE_API_BASE=XXXXXX
python gen_gpt_answer.py # This writes to data/japanese_safety_bench/model_answer/gpt-4.jsonl

# Convert the output to a format that can be used by SFT/DPO code
# Refer to https://www.notion.so/stabilityai/How-to-run-SFT-DPO-26a4108de6aa4c80bb9463e49919052b?pvs=4#12190dad152a4c9ebf3e971caee38080
jq -c '{id: .question_id|tostring, messages: [{content: .instruction, role: "human"}, {content: .response, role: "assistant"}]}' data/japanese_safety_bench/model_answer/gpt-4.jsonl > data/japanese_safety_bench/model_answer/gpt-4-for-sft.jsonl
```

Finally, convert this to a Hugging Face dataset via the following:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "data/japanese_safety_bench/model_answer/gpt-4-for-sft.jsonl"})['train']
ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
ds.save_to_disk("/path/to/processed/do-not-answer-ja-gpt4")
```

### Running instruction tuning

```bash
git clone git@github.com:Stability-AI/jp-alignment-handbook.git
git checkout dev/fine-tuning-safety

python3.10 -mvenv .venv
source .venv/bin/activate
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install .
pip install flash-attn==2.3.6 --no-build-isolation

export WANDB_API_KEY=UPDATE_ME
export EXP_NAME=jslm2-1_6b-jp-sft-v1_13_1_safety
sbatch run.slurm
```

## Generating `question.jsonl`

`question.jsonl` was generated via the following.

First, convert the Hugging Face dataset to jsonl:

```python
from datasets import load_dataset

data = load_dataset("kunishou/do-not-answer-ja", split="train")
data.to_json("do-not-answer-ja.jsonl")
```

Then, reorganize the jsonl so that it matches FastChat's format, using jq:

```bash
jq -c '{question_id: (.id)|tonumber, category: .risk_area, turns: [.question], types_of_harm: .types_of_harm, specific_harms: .specific_harms}' do-not-answer-ja.jsonl > question.jsonl
```
