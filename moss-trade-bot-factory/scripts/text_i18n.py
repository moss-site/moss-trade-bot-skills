"""Helpers for validating bilingual bot text payloads."""


def _clean(value):
    return str(value or "").strip()


def default_text(value):
    zh = _clean((value or {}).get("zh", ""))
    if zh:
        return zh
    return _clean((value or {}).get("en", ""))


def contains_han(value):
    return any("\u4e00" <= ch <= "\u9fff" for ch in _clean(value))


def validate_bilingual_text(field, value, max_len):
    zh = _clean((value or {}).get("zh", ""))
    en = _clean((value or {}).get("en", ""))
    if not zh or not en:
        raise ValueError(f"{field}.zh and {field}.en are required")
    if len(zh) > max_len:
        raise ValueError(f"{field}.zh too long (max {max_len})")
    if len(en) > max_len:
        raise ValueError(f"{field}.en too long (max {max_len})")
    if not contains_han(zh):
        raise ValueError(f"{field}.zh must contain Chinese text")
    if contains_han(en):
        raise ValueError(f"{field}.en must be English text without Chinese characters")
    return {"zh": zh, "en": en}
