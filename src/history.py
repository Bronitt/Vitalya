from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from logger import log

# root_project/.history/<chat_name>.json — .history.py лежит в корне проекта,
# так что родительская папка этого файла и есть root_project.
HISTORY_DIR: Path = Path(__file__).resolve().parent.parent / ".history"

_BAD_CHARS = re.compile(r'[\\/:*?"<>|]+')
DEFAULT_TITLE = "Новый чат"


def _slugify(title: str, maxlen: int = 60) -> str:
    # Только символы, недопустимые в имени файла на Windows/Linux, остальное
    # (кириллицу, пробелы, эмодзи) оставляем как есть — имя файла = имя чата.
    slug = _BAD_CHARS.sub("_", title).strip().strip(".")
    slug = re.sub(r"\s+", " ", slug)
    return slug[:maxlen] or DEFAULT_TITLE


def derive_title(first_message: str, maxlen: int = 40) -> str:
    # Автоназвание чата по первому сообщению — как делают современные чат-интерфейсы.
    line = (first_message or "").strip().splitlines()[0].strip() if first_message.strip() else ""
    if len(line) > maxlen:
        line = line[:maxlen].rstrip() + "…"
    return line or DEFAULT_TITLE


def unique_path(title: str) -> Path:
    """Путь для НОВОГО чата: root_project/.history/<slug(title)>.json,
    с дописыванием " (2)", " (3)"... при совпадении имени."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    base = _slugify(title)
    path = HISTORY_DIR / f"{base}.json"
    n = 2
    while path.exists():
        path = HISTORY_DIR / f"{base} ({n}).json"
        n += 1
    return path


@dataclass
class ChatFile:
    path: Path
    title: str
    updated_at: float

    @property
    def chat_id(self) -> str:
        return self.path.stem


def list_chats() -> list[ChatFile]:
    """Все сохранённые чаты, отсортированные от свежих к старым."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    chats: list[ChatFile] = []
    for p in sorted(HISTORY_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("пропускаю повреждённый файл истории: %s", p)
            continue
        chats.append(ChatFile(
            path=p,
            title=data.get("title") or p.stem,
            updated_at=data.get("updated_at", 0.0),
        ))
    chats.sort(key=lambda c: c.updated_at, reverse=True)
    return chats


def load_chat(path: Path) -> tuple[str, list[dict]]:
    """Возвращает (заголовок, сообщения без системного промпта)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("title") or Path(path).stem, data.get("messages", [])


def save_chat(path: Path, title: str, messages: list[dict]) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    data = {"title": title, "updated_at": time.time(), "messages": messages}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")