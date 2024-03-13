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
