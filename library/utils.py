import re

def json_to_pseudo_text(json_data):
    """
    Преобразует JSON-объекты в псевдоформат вида:
    объект1 (признак1 признак2) объект2 ()
    """
    objects = []
    for obj_dict in json_data:
        for obj_name, attrs in obj_dict.items():
            attr_text = " ".join(attrs) if attrs else ""
            objects.append(f"{obj_name} ({attr_text})")
    return " ".join(objects)

def pseudo_text_to_json(text):
    """
    Парсит строку вида 'объект1 (пр1 пр2) объект2 ()' в список объектов с признаками.
    Если формат нарушен — возвращает [].
    """
    try:
        result = []
        pattern = re.findall(r'(\S+)\s*\(([^()]*)\)', text)
        for obj_name, attrs_str in pattern:
            attrs = [attr.strip() for attr in attrs_str.strip().split() if attr.strip()]
            result.append({obj_name: attrs})
        return result
    except Exception as e:
        print(f"Ошибка при разборе псевдокода: {e}")
        return []
