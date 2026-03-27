from __future__ import annotations

from typing import Any, Dict


def convert_to_int(val: Any) -> Any:
    try:
        int(val)
        return str(int(val))
    except Exception:
        pass

    try:
        float_val = float(val)
        if float_val == int(float_val):
            return str(int(float_val))
        return val
    except Exception:
        return val


def preprocess(value: Any, question_key: str, preprocess_map: Dict[str, Dict[str, str]]) -> Any:
    value = convert_to_int(value)
    if question_key not in preprocess_map or str(value) not in preprocess_map[question_key]:
        return value

    ret_val = preprocess_map[question_key][str(value)]
    return int(ret_val) if ret_val else ""


def process(
    value: Any,
    question_key: str,
    question_metadata: Dict[str, Any],
    codebook: Dict[str, Dict[str, Any]],
) -> Any:
    ques_info = question_metadata[question_key]
    if value == "":
        return ""
    if ques_info["answer_data_type"] == "non_ordinal":
        if str(value) in codebook[question_key]["choices"]:
            return codebook[question_key]["choices"][str(value)]
        return value
    if ques_info["answer_data_type"] == "ordinal":
        value = int(value)
        if value < 0:
            return ""
        min_range = ques_info["answer_scale_min"]
        max_range = ques_info["answer_scale_max"]
        if value < min_range or value > max_range:
            raise ValueError(
                f"Value {value} for {question_key} fell outside [{min_range}, {max_range}]"
            )
        return value
    return value
