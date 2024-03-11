
`question.jsonl` was generated via the following:

```python
from datasets import load_dataset

data = load_dataset("kunishou/do-not-answer-ja", split="train")
data.to_json("do-not-answer-ja.jsonl")
```

```bash
jq -c '{question_id: (.id)|tonumber, category: .risk_area, turns: [.question], types_of_harm: .types_of_harm, specific_harms: .specific_harms}' do-not-answer-ja.jsonl > question.jsonl
```