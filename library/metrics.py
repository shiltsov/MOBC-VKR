from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
import spacy

# Загружаем русскую модель spaCy
nlp = spacy.load("ru_core_news_sm")

def lemmatize_spacy(text: str) -> str:
    """
    Лемматизирует отдельное слово или фразу (берёт первую лемму из анализа spaCy).
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def normalize_scene_json_spacy(scene_json: Dict) -> Dict:
    """
    Возвращает новый JSON, где все объекты и признаки лемматизированы через spaCy.
    """
    normalized_scene = {"scene": {"location": scene_json["scene"].get("location", ""), "objects": []}}
    for obj_dict in scene_json["scene"]["objects"]:
        for obj_name, attributes in obj_dict.items():
            obj_lemma = lemmatize_spacy(obj_name)
            attr_lemmas = [lemmatize_spacy(attr) for attr in attributes]
            normalized_scene["scene"]["objects"].append({obj_lemma: attr_lemmas})
    return normalized_scene

def extract_objects_and_attributes(scene_json: Dict) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Извлекает список объектов и список (объект, признак) пар из JSON сцены.
    """
    objects = []
    object_attributes = []
    for obj_dict in scene_json["scene"]["objects"]:
        for obj, attrs in obj_dict.items():
            objects.append(obj)
            for attr in attrs:
                object_attributes.append((obj, attr))
    return objects, object_attributes


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Вычисляет precision, recall и F1.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_scene_extraction(gt_json: Dict, pred_json: Dict) -> Dict[str, float]:
    # Извлекаем объекты и признаки
    gt_objects, gt_pairs = extract_objects_and_attributes(gt_json)
    pred_objects, pred_pairs = extract_objects_and_attributes(pred_json)

    # F1 по объектам
    gt_object_set = set(gt_objects)
    pred_object_set = set(pred_objects)
    tp_obj = len(gt_object_set & pred_object_set)
    fp_obj = len(pred_object_set - gt_object_set)
    fn_obj = len(gt_object_set - pred_object_set)
    _, _, f1_object = precision_recall_f1(tp_obj, fp_obj, fn_obj)

    # F1 по (объект, атрибут) парам
    gt_pair_set = set(gt_pairs)
    pred_pair_set = set(pred_pairs)
    tp_attr = len(gt_pair_set & pred_pair_set)
    fp_attr = len(pred_pair_set - gt_pair_set)
    fn_attr = len(gt_pair_set - pred_pair_set)
    _, _, f1_attr = precision_recall_f1(tp_attr, fp_attr, fn_attr)

    # Взвешенное объединение
    f1_combined_weighted = (
        (len(gt_object_set) * f1_object + len(gt_pair_set) * f1_attr)
        / (len(gt_object_set) + len(gt_pair_set))
        if (len(gt_object_set) + len(gt_pair_set)) > 0
        else 0.0
    )
    
    # Простое среднее
    f1_combined_simple = (f1_object + f1_attr) / 2

    return {
        "f1_object": f1_object,
        "f1_attribute": f1_attr,
        "f1_combined_weighted": f1_combined_weighted,
        "f1_combined_simple": f1_combined_simple,
    }

def evaluate_global_f1_on_pairs(gt_json: Dict, pred_json: Dict) -> Dict[str, float]:
    """
    Вычисляет глобальный Precision, Recall и F1 по всем (объект, признак) парам.
    """
    _, gt_pairs = extract_objects_and_attributes(gt_json)
    _, pred_pairs = extract_objects_and_attributes(pred_json)

    gt_pair_set = set(gt_pairs)
    pred_pair_set = set(pred_pairs)

    tp = len(gt_pair_set & pred_pair_set)
    fp = len(pred_pair_set - gt_pair_set)
    fn = len(gt_pair_set - pred_pair_set)

    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    return {
        "global_precision": precision,
        "global_recall": recall,
        "global_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }




