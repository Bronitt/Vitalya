from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from logger import log

# --- 1. системные утилиты: одинаковы на любой машине, хардкод ок ---------
SYSTEM_APPS: dict[str, str] = {
    "блокнот": "notepad.exe",
    "калькулятор": "calc.exe",
    "проводник": "explorer.exe",
}

# --- 2. сторонние программы: разрешаем ИМЕНА, путь ищем в системе --------
THIRD_PARTY_APPS: dict[str, str] = {

}

# ручной оверрайд на случай portable-версий / нестандартной установки
OVERRIDE_PATH: Path = Path(__file__).resolve().parent / "apps_whitelist.json"


def _from_windows_app_paths(exe_name: str) -> str | None:
    """Спрашиваем у ОС, куда установлена программа — без хардкода и без файла.
    Работает для подавляющего большинства инсталляторов на Windows."""
    if sys.platform != "win32":
        return None
    import winreg
    key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, None)  # (по умолчанию) = путь
                if value and Path(value).is_file():
                    return value
        except OSError:
            continue
    return None


def _load_overrides() -> dict[str, str]:
    if not OVERRIDE_PATH.exists():
        return {}
    try:
        return json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.error("не удалось прочитать %s: %s", OVERRIDE_PATH, e)
        return {}


_cache: dict[str, str] | None = None


def load_whitelist(force_reload: bool = False) -> dict[str, str]:
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    overrides = _load_overrides()
    resolved: dict[str, str] = {}

    # системные — просто через PATH
    for name, exe in SYSTEM_APPS.items():
        found = shutil.which(exe)
        if found:
            resolved[name] = found
        else:
            log.warning("системная утилита %r (%s) не найдена в PATH", name, exe)

    # сторонние — сначала система, потом ручной оверрайд, потом PATH как последний шанс
    for name, exe in THIRD_PARTY_APPS.items():
        path = (
            overrides.get(name)
            or _from_windows_app_paths(exe)
            or shutil.which(exe)
        )
        if path and Path(path).is_file():
            resolved[name] = path
        else:
            log.warning(
                "программа %r не найдена автоматически; если она установлена, "
                "пропишите путь вручную в %s", name, OVERRIDE_PATH.name,
            )

    _cache = resolved
    return _cache