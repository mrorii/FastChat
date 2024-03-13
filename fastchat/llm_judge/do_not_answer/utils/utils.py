# Adapted from https://github.com/Libr-AI/do-not-answer/blob/30ae028/do_not_answer/utils/utils.py
import json
from typing import Dict, Any


def save_json(dictionary: Dict[str, Any], save_dir: str) -> None:
    # Serializing json
    json_object = json.dumps(dictionary, indent=4, ensure_ascii=False)

    # Writing to sample.json
    with open(save_dir, "w", encoding='utf-8') as outfile:
        outfile.write(json_object)


def read_json(filepath: str) -> Dict[str, Any]:
    data = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
