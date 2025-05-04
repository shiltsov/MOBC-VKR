# Метрики, используемые при обучении моделей

import spacy
from typing import List, Dict, Tuple
from collections import Counter

# Загружаем модель spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

def lemmatize_phrase_spacy(phrase: str) -> str:
    """Лемматизирует фразу с помощью spaCy (только леммы, без пунктуации)."""
    doc = nlp(phrase.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

def normalize_structure(data: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Преобразует список словарей в один словарь с лемматизированными объектами и признаками."""
    result = {}
    for obj_dict in data:
        for obj, attrs in obj_dict.items():
            obj_lemma = lemmatize_phrase_spacy(obj)
            attr_lemmas = [lemmatize_phrase_spacy(attr) for attr in attrs]
            result.setdefault(obj_lemma, []).extend(attr_lemmas)
    # Удалим дубли
    result = {k: list(set(v)) for k, v in result.items()}
    return result

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate_all_metrics(pred_raw: List[Dict[str, List[str]]], label_raw: List[Dict[str, List[str]]]) -> Dict[str, float]:
    pred = normalize_structure(pred_raw)
    label = normalize_structure(label_raw)

    # F1 по объектам
    pred_objects = set(pred.keys())
    label_objects = set(label.keys())
    tp_obj = len(pred_objects & label_objects)
    fp_obj = len(pred_objects - label_objects)
    fn_obj = len(label_objects - pred_objects)
    _, _, f1_objects = precision_recall_f1(tp_obj, fp_obj, fn_obj)

    # F1 по признакам по каждому объекту (усреднение по объектам, macro)
    f1_per_object = []
    total_attrs = 0
    weighted_sum = 0
    for obj in label_objects | pred_objects:
        label_attrs = set(label.get(obj, []))
        pred_attrs = set(pred.get(obj, []))
        tp = len(label_attrs & pred_attrs)
        fp = len(pred_attrs - label_attrs)
        fn = len(label_attrs - pred_attrs)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        f1_per_object.append(f1)
        weighted_sum += f1 * len(label_attrs)
        total_attrs += len(label_attrs)

    f1_attributes_macro = sum(f1_per_object) / len(f1_per_object) if f1_per_object else 0.0
    f1_attributes_weighted = weighted_sum / total_attrs if total_attrs > 0 else 0.0

    # Глобальный F1 по парам (obj, attr)
    pred_pairs = {(obj, attr) for obj, attrs in pred.items() for attr in attrs}
    label_pairs = {(obj, attr) for obj, attrs in label.items() for attr in attrs}
    tp_pairs = len(pred_pairs & label_pairs)
    fp_pairs = len(pred_pairs - label_pairs)
    fn_pairs = len(label_pairs - pred_pairs)
    _, _, f1_global_pairs = precision_recall_f1(tp_pairs, fp_pairs, fn_pairs)

    # Объединённые метрики
    f1_combined_simple = (f1_objects + f1_attributes_macro) / 2
    total_obj = len(label_objects)
    f1_combined_weighted = ((total_obj * f1_objects) + (total_attrs * f1_attributes_weighted)) / (total_obj + total_attrs) if (total_obj + total_attrs) > 0 else 0.0

    return {
        "f1_objects": round(f1_objects, 4),
        "f1_attributes_macro": round(f1_attributes_macro, 4),
        "f1_attributes_weighted": round(f1_attributes_weighted, 4),
        "f1_global_obj_attr_pairs": round(f1_global_pairs, 4),
        "f1_combined_simple": round(f1_combined_simple, 4),
        "f1_combined_weighted": round(f1_combined_weighted, 4),
    }


## метрики для spacial relations
def evaluate_relation_predictions(data):
    """
    Эта метрика только для обучения угадывать связи между объектами
    """

    true_binary = [0 if item["target"] == "нет связи" else 1 for item in data]
    pred_binary = [0 if item["predicted_target"] == "нет связи" else 1 for item in data]

    # Binary F1 вручную чтобы не грузить громоздкий sklearn
    TP = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    FP = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    FN = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)

    precision_binary = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_binary = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_binary = (2 * precision_binary * recall_binary) / (precision_binary + recall_binary) if (precision_binary + recall_binary) > 0 else 0.0

    # Strict F1 (точное совпадение метки связи)
    TP_strict = sum(1 for item in data if item["target"] == item["predicted_target"])
    FP_strict = len(data) - TP_strict  # каждая пара — один шанс на TP
    FN_strict = FP_strict  # аналогично

    precision_strict = TP_strict / (TP_strict + FP_strict) if (TP_strict + FP_strict) > 0 else 0.0
    recall_strict = TP_strict / (TP_strict + FN_strict) if (TP_strict + FN_strict) > 0 else 0.0
    f1_strict = (2 * precision_strict * recall_strict) / (precision_strict + recall_strict) if (precision_strict + recall_strict) > 0 else 0.0

    return {
        "F1binary": round(f1_binary, 4),
        "F1strict": round(f1_strict, 4),
    }

### 

