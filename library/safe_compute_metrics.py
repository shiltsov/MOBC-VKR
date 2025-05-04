import json
from typing import List, Dict, Any

from .metrics import evaluate_scene_extraction  

def safe_parse_json(text: str) -> Any:
    """Попытаться безопасно распарсить JSON, вернуть None в случае ошибки."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def safe_compute_metrics(predictions: List[str], labels: List[str]) -> Dict[str, float]:
    """
    Безопасный подсчёт метрик на предсказаниях модели.
    predictions: список строк (предсказаний модели)
    labels: список строк (эталонных правильных ответов)
    """

    total = len(predictions)
    valid = 0
    f1_object_list = []
    f1_attribute_list = []
    f1_combined_weighted_list = []
    f1_combined_simple_list = []

    for pred_text, label_text in zip(predictions, labels):
        pred_json = safe_parse_json(pred_text)
        label_json = safe_parse_json(label_text)

        if pred_json is None or label_json is None:
            # Невалидный JSON → 0 по всем метрикам
            f1_object_list.append(0.0)
            f1_attribute_list.append(0.0)
            f1_combined_weighted_list.append(0.0)
            f1_combined_simple_list.append(0.0)
        else:
            valid += 1
            scores = evaluate_scene_extraction(label_json, pred_json)
            f1_object_list.append(scores["f1_object"])
            f1_attribute_list.append(scores["f1_attribute"])
            f1_combined_weighted_list.append(scores["f1_combined_weighted"])
            f1_combined_simple_list.append(scores["f1_combined_simple"])

    return {
        "f1_object": round(sum(f1_object_list) / total, 4),
        "f1_attribute": round(sum(f1_attribute_list) / total, 4),
        "f1_combined_weighted": round(sum(f1_combined_weighted_list) / total, 4),
        "f1_combined_simple": round(sum(f1_combined_simple_list) / total, 4),
        "valid_json_rate": round(valid / total, 4),
        "total_samples": total,
        "valid_samples": valid,
    }
