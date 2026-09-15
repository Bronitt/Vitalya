import subprocess
from pathlib import Path
from typing import Callable

from agent.apps import load_whitelist
from agent.winctl import type_into_app
from state import Context


TOOL_REGISTRY: dict[str, Callable] = {}
TOOL_SCHEMAS: list[dict] = []


def tool(name: str, description: str, params: dict):
    # Регистрирует функцию как инструмент и собирает схему для Ollama.
    def wrap(fn):
        TOOL_REGISTRY[name] = fn
        TOOL_SCHEMAS.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": params},
        })
        return fn
    return wrap


def safe_path(ctx: Context, raw: str) -> Path:
    # Пускаем только внутрь песочницы. Единственная точка проверки путей.
    target = (ctx.sandbox / raw).resolve()
    if not str(target).startswith(str(ctx.sandbox)):
        raise PermissionError(f"Путь вне песочницы: {raw}")
    return target


@tool(
    "list_files",
    "Показать файлы в рабочей папке пользователя",
    {"type": "object", "properties": {"subdir": {"type": "string"}}, "required": []},
)
def list_files(ctx: Context, subdir: str = ".") -> str:
    p = safe_path(ctx, subdir)
    if not p.is_dir():
        return f"Папки {subdir} не существует"
    names = [f.name for f in sorted(p.iterdir())][:20]
    return ", ".join(names) if names else "Папка пуста"


@tool(
    "create_note",
    "Создать текстовый файл с заметкой",
    {
        "type": "object",
        "properties": {"filename": {"type": "string"}, "text": {"type": "string"}},
        "required": ["filename", "text"],
    },
)
def create_note(ctx: Context, filename: str, text: str) -> str:
    p = safe_path(ctx, filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return f"Создан файл {p.name}, {len(text)} символов"


@tool(
    "open_app",
    "Открыть разрешённое приложение на компьютере",
    {"type": "object", "properties": {"app": {"type": "string"}}, "required": ["app"]},
)
def open_app(ctx: Context, app: str) -> str:
    key = app.lower().strip()
    allowed = load_whitelist()
    exe = allowed.get(key)
    if not exe:
        return f"Приложение {app} не в списке разрешённых"
    proc = subprocess.Popen([exe], shell=False)
    ctx.running_apps[key] = proc  # запоминаем, чтобы потом можно было туда печатать
    return f"Открываю {app}"


@tool(
    "type_text",
    "Напечатать текст в уже открытом приложении (сначала вызови open_app)",
    {
        "type": "object",
        "properties": {"app": {"type": "string"}, "text": {"type": "string"}},
        "required": ["app", "text"],
    },
)
def type_text(ctx: Context, app: str, text: str) -> str:
    key = app.lower().strip()
    proc = ctx.running_apps.get(key)
    if proc is None:
        return f"{app} не запущен агентом — сначала вызови open_app"
    if proc.poll() is not None:
        ctx.running_apps.pop(key, None)
        return f"{app} уже закрыт"
    try:
        type_into_app(proc.pid, text)
    except RuntimeError as e:
        return f"Не удалось напечатать текст: {e}"
    return f"Текст напечатан в {app}"