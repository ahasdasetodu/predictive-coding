from collections.abc import Mapping
from pathlib import Path

import yaml

_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def string_key_mapping(loaded: object, what: str) -> dict[str, object]:
    """Copy a YAML mapping into a dict with string keys."""
    if not isinstance(loaded, dict):
        raise ValueError(f"{what} must be a mapping")
    mapping: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError(f"{what} keys must be strings")
        mapping[key] = value
    return mapping


def reject_unknown(mapping: Mapping[str, object], allowed: set[str], what: str) -> None:
    """Raise when a mapping contains a key this format does not define."""
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {what} keys: {', '.join(unknown)}")


def require(mapping: Mapping[str, object], required: set[str], what: str) -> None:
    """Raise when a mapping omits a key this format needs."""
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"missing {what} keys: {', '.join(missing)}")


def run_name(value: object) -> str:
    """A filesystem-safe run name."""
    if not isinstance(value, str) or not value or not value[0].isalnum():
        raise ValueError("name must be a filesystem-safe token")
    if any(character not in _NAME_CHARS for character in value):
        raise ValueError("name must be a filesystem-safe token")
    return value


def as_int(value: object, key: str) -> int:
    """An int. Booleans are rejected because they are ints in Python."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an int")
    return value


def at_least(value: int, key: str, minimum: int) -> int:
    """`value` when it is at least `minimum`."""
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def as_float(value: object, key: str) -> float:
    """A float. Integers from YAML are accepted; booleans are not."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a float")
    return float(value)


def optional_float(raw: Mapping[str, object], key: str, default: float) -> float:
    """A float from `raw`, or `default` when the key is omitted."""
    if key not in raw:
        return default
    return as_float(raw[key], key)


def negative_float(value: object, key: str) -> float:
    """A float below zero, so a variance of one stays reachable."""
    number = as_float(value, key)
    if number >= 0.0:
        raise ValueError(f"{key} must be negative")
    return number


def as_bool(value: object, key: str) -> bool:
    """A boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a bool")
    return value


def as_text(value: object, key: str) -> str:
    """A non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def render_yaml(document: dict[str, object]) -> str:
    """Dump one mapping. The caller checks that parsing it returns the same data."""
    rendered = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    if not isinstance(rendered, str):
        raise TypeError("YAML dump did not return text")
    return rendered


def write_yaml(path: Path, document: dict[str, object]) -> None:
    """Replace a YAML file with `document`."""
    write_text(path, render_yaml(document))


def write_text(path: Path, text: str) -> None:
    """Replace a file with `text`."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
