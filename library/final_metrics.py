# все сцены имеют формат
#
#{
#    "scene": {
#        "location": "автосервис",
#        "objects": [
#            {"гаечный ключ": ["металлический"]},
#            {"домкрат": ["металлический", "тяжелый", "прочный"]},
#            {"аккумулятор": []}
#        ],
#        "relations": [
#            ["аккумулятор", "рядом с", "гаечный ключ"],
#            ["гаечный ключ", "на", "домкрат"]
#        ]
#    }
#}

import spacy
import networkx as nx

from typing import List, Dict, Tuple
from collections import Counter
from networkx.algorithms.similarity import optimize_graph_edit_distance
from typing import Dict

nlp = spacy.load("ru_core_news_sm")

def lemmatize_scene(scene: Dict, nlp) -> Dict:
    def lemmatize(text: str) -> str:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    new_scene = {
        "scene": {
            "location": scene["scene"].get("location", "неизвестно"),
            "objects": [],
            "relations": []
        }
    }

    # Лемматизируем объекты и их признаки
    for obj in scene["scene"].get("objects", []):
        for obj_name, attributes in obj.items():
            obj_lemma = lemmatize(obj_name)
            attrs_lemma = [lemmatize(attr) for attr in attributes]
            new_scene["scene"]["objects"].append({obj_lemma: attrs_lemma})

    # Лемматизируем связи
    for subj, rel, obj in scene["scene"].get("relations", []):
        subj_lemma = lemmatize(subj)
        rel_lemma = lemmatize(rel)
        obj_lemma = lemmatize(obj)
        new_scene["scene"]["relations"].append([subj_lemma, rel_lemma, obj_lemma])

    return new_scene

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate_obj_attr_metrics(pred, label, normalize = True) -> Dict[str, float]:
    
    # лемматизируем если нужно
    if normalize:
        nlp = spacy.load("ru_core_news_sm")
        pred = lemmatize_scene(pred,nlp)
        label = lemmatize_scene(label,nlp)

    # F1 по объектам
    #print(pred["scene"]["objects"], label["scene"]["objects"])
    
    pred_objects = set([list(descr.keys())[0] for descr in  pred["scene"]["objects"]])
    label_objects = set([list(descr.keys())[0] for descr in  label["scene"]["objects"]])
    
    #print(pred_objects, label_objects)
    # до сих пор ок
    
    tp_obj = len(pred_objects & label_objects)
    fp_obj = len(pred_objects - label_objects)
    fn_obj = len(label_objects - pred_objects)
    _, _, f1_objects = precision_recall_f1(tp_obj, fp_obj, fn_obj)

    # F1 по признакам по каждому объекту (усреднение по объектам, macro)
    f1_per_object = []
    total_attrs = 0
    weighted_sum = 0

    for obj in label_objects | pred_objects:
        # по объекту из пересечения извлекаем атрибуты 
        # ну так себе конечно, можно и переписать но в целом ок, хотя и легаст
        # суть в том что в одном из объекта может не быть - тогда и атрибутов нет
        try:
            label_attrs  = set([obj_dict[obj] for obj_dict in label["scene"]["objects"] if obj in obj_dict.keys()][0])
        except:
            label_attrs = set()
        try:    
            pred_attrs  = set([obj_dict[obj] for obj_dict in pred["scene"]["objects"] if obj in obj_dict.keys()][0])
        except:
            pred_attrs = set()
                    
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
        "f1_combined_weighted": round(f1_combined_weighted, 4)
    }

def scene_to_graph_with_attrs(scene: Dict) -> nx.DiGraph:
    G = nx.DiGraph()

    # Вершины: объекты и атрибуты
    for obj in scene["scene"]["objects"]:
        for obj_name, attributes in obj.items():
            G.add_node(obj_name, type="object")
            for attr in attributes:
                G.add_node(attr, type="attribute")
                G.add_edge(obj_name, attr, relation=None)

    # Пространственные отношения
    for subj, rel, obj in scene["scene"]["relations"]:
        G.add_edge(subj, obj, relation=rel)

    return G

# Функция сопоставления меток узлов
# узлы совпадают если в вершине одинаковые объекты
def node_match(n1, n2):
    return n1.get("type") == n2.get("type")


# Функция сопоставления меток рёбер
# если у обоих relation=None — считается совпадением (нормально)
# то есть ребра объект-атрибут в обоих гафах считаются совпадающими даже при условии что нет метки
def edge_match(e1, e2):
    return e1.get("relation") == e2.get("relation")


def evaluate_ged_score(scene1: Dict, scene2: Dict, normalize = True) -> float:
    if normalize:
        nlp = spacy.load("ru_core_news_sm")
        scene1 = lemmatize_scene(scene1, nlp)
        scene2 = lemmatize_scene(scene2, nlp)    
    
    G1 = scene_to_graph_with_attrs(scene1)
    G2 = scene_to_graph_with_attrs(scene2)

    
    # Оптимизируем GED с учётом атрибутов
    # optimize_graph_edit_distance использует итеративный эвристический поиск 
    # по возможным сопоставлениям графов, первая выдача как правило оптимальная
    # (но NP сложная задача и на больших графах работает медленно)
    ged_iter = optimize_graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match)

    try:
        edit_distance = next(ged_iter)
    except StopIteration:
        return 0.0

    # Нормализация: максимум — если ни одной вершины и ребра не совпало
    max_size = max(G1.number_of_nodes() + G1.number_of_edges(),
                   G2.number_of_nodes() + G2.number_of_edges())

    if max_size == 0:
        return 1.0  # два пустых графа считаем полностью совпадающими

    similarity = 1.0 - (edit_distance / max_size)
    return {
        "GED_score" : round(similarity, 4)
    }
