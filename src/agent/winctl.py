
from __future__ import annotations

import sys
import time

_IS_WINDOWS = sys.platform == "win32"


def _child_pids(root_pid: int) -> set[int]:
    """PID процесса + всех его потомков (нужно для приложений вроде VS Code,
    которые запускают реальное окно из другого процесса, чем тот, что вернул
    subprocess.Popen). Требует psutil; если его нет — работаем только с root_pid
    (этого достаточно для простых программ типа блокнота/калькулятора)."""
    pids = {root_pid}
    try:
        import psutil
        try:
            parent = psutil.Process(root_pid)
            pids.update(c.pid for c in parent.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    except ImportError:
        pass  # psutil не обязателен, деградируем до одного pid
    return pids


def find_window_by_pid(root_pid: int, timeout: float = 5.0) -> int | None:
    """Ищет видимое top-level окно с непустым заголовком среди процесса
    и его потомков. Приложение может открываться не мгновенно — поэтому
    опрашиваем в цикле до timeout секунд."""
    if not _IS_WINDOWS:
        raise RuntimeError("управление окнами реализовано только для Windows")

    import win32gui
    import win32process

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = _child_pids(root_pid)
        found: list[int] = []

        def _cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd) or not win32gui.GetWindowText(hwnd):
                return
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid in pids:
                found.append(hwnd)

        win32gui.EnumWindows(_cb, None)
        if found:
            return found[0]
        time.sleep(0.2)

    return None


def focus_window(hwnd: int, retries: int = 3) -> bool:
    import win32con
    import win32gui

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    for _ in range(retries):
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            # Windows иногда блокирует SetForegroundWindow из фонового процесса —
            # обходной путь через прикрепление к потоку окна ввода.
            import win32api
            import win32process

            fg_hwnd = win32gui.GetForegroundWindow()
            fg_thread, _ = win32process.GetWindowThreadProcessId(fg_hwnd)
            target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
            cur_thread = win32api.GetCurrentThreadId()
            win32process.AttachThreadInput(cur_thread, fg_thread, True)
            win32process.AttachThreadInput(cur_thread, target_thread, True)
            try:
                win32gui.SetForegroundWindow(hwnd)
            finally:
                win32process.AttachThreadInput(cur_thread, fg_thread, False)
                win32process.AttachThreadInput(cur_thread, target_thread, False)

        time.sleep(0.15)
        if win32gui.GetForegroundWindow() == hwnd:
            return True
    return False


def paste_text(text: str) -> None:
    """Кладёт text в буфер обмена и посылает Ctrl+V активному окну."""
    import win32clipboard
    import win32con
    import win32api

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()

    time.sleep(0.05)  # дать окну реально получить фокус перед вводом
    win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
    win32api.keybd_event(ord("V"), 0, 0, 0)
    win32api.keybd_event(ord("V"), 0, win32con.KEYEVENTF_KEYUP, 0)
    win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)


def type_into_app(pid: int, text: str) -> None:
    hwnd = find_window_by_pid(pid)
    if hwnd is None:
        raise RuntimeError("окно приложения не появилось за отведённое время")
    if not focus_window(hwnd):
        raise RuntimeError("не удалось передать фокус окну приложения")
    paste_text(text)