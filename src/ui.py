from __future__ import annotations

import asyncio
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import TYPE_CHECKING

import history
from state import State

if TYPE_CHECKING:
    from engine import Assistant


STATE_LABELS: dict[State, str] = {
    State.BOOT: "Загрузка моделей...",
    State.IDLE: "Готов. Зажми кнопку и говори (или напиши текст)",
    State.LISTENING: "Слушаю...",
    State.THINKING: "Думаю...",
    State.ACTING: "Выполняю действие...",
    State.SPEAKING: "Говорю... (можно нажать кнопку, чтобы прервать)",
    State.ERROR: "Ошибка, восстанавливаюсь",
    State.SHUTDOWN: "Завершение...",
}

STATE_COLORS: dict[State, str] = {
    State.BOOT: "#757575",
    State.IDLE: "#2e7d32",
    State.LISTENING: "#c62828",
    State.THINKING: "#f9a825",
    State.ACTING: "#1565c0",
    State.SPEAKING: "#6a1b9a",
    State.ERROR: "#b71c1c",
    State.SHUTDOWN: "#424242",
}


class AssistantUI:
    """Минимальное окно поверх Assistant.

    Работает в одном потоке с asyncio-циклом: вместо блокирующего
    root.mainloop() приложение вызывает root.update() внутри корутины
    tick_forever() на каждом тике event loop. Поэтому обработчики
    tkinter выполняются на потоке event loop и могут напрямую
    выставлять asyncio.Event — без call_soon_threadsafe, как это было
    нужно для потока keyboard-хуков.
    """

    def __init__(self, app: "Assistant") -> None:
        self.app = app
        self._closed = False

        char_name = self.app.engines.cfg.char.char_name
        # Имена ролей в логе — по конфигу персонажа, а не хардкодом.
        self.role_prefix = {"user": "Ты", "assistant": char_name, "tool": "Инструмент"}

        self.root = tk.Tk()
        self.root.title(char_name)
        self.root.geometry("440x560")
        self.root.minsize(360, 460)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda _e: self._on_close())

        # ---- панель чатов: список сохранённых + кнопка нового ----
        chat_frame = tk.Frame(self.root)
        chat_frame.pack(fill="x", padx=8, pady=(8, 0))
        self._chat_options: list[tuple[str, Path]] = []
        self.chat_combo = ttk.Combobox(chat_frame, state="readonly")
        self.chat_combo.pack(side="left", fill="x", expand=True)
        self.chat_combo.bind("<<ComboboxSelected>>", self._on_chat_selected)
        self.new_chat_btn = tk.Button(chat_frame, text="+ Новый чат", command=self._on_new_chat)
        self.new_chat_btn.pack(side="left", padx=(6, 0))
        self._refresh_chat_list()

        self.status_var = tk.StringVar(value=STATE_LABELS[State.BOOT])
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, font=("Segoe UI", 11, "bold"),
            fg="white", bg=STATE_COLORS[State.BOOT], pady=8, wraplength=420, justify="center",
        )
        self.status_label.pack(fill="x", pady=(8, 0))

        self.log_widget = scrolledtext.ScrolledText(
            self.root, state="disabled", wrap="word", font=("Segoe UI", 10),
        )
        self.log_widget.pack(fill="both", expand=True, padx=8, pady=8)

        entry_frame = tk.Frame(self.root)
        entry_frame.pack(fill="x", padx=8, pady=(0, 4))
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(entry_frame, textvariable=self.entry_var)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._on_send_text)
        self.send_btn = tk.Button(entry_frame, text="Отправить", command=self._on_send_text)
        self.send_btn.pack(side="left", padx=(6, 0))

        self.talk_btn = tk.Button(
            self.root, text="Зажми и говори", font=("Segoe UI", 12), height=2,
        )
        self.talk_btn.pack(fill="x", padx=8, pady=(0, 8))
        self.talk_btn.bind("<ButtonPress-1>", self._on_press_talk)
        self.talk_btn.bind("<ButtonRelease-1>", self._on_release_talk)

    # ---- подписки на Context (state.py) ----

    def on_state_change(self, _old: State, new: State) -> None:
        self.status_var.set(STATE_LABELS.get(new, new.value))
        self.status_label.configure(bg=STATE_COLORS.get(new, "#333333"))
        # Переключать чат посреди хода диалога нельзя — Assistant всё равно
        # проигнорирует вызов, но лучше сразу не давать на это нажать.
        idle = new is State.IDLE
        self.new_chat_btn.configure(state="normal" if idle else "disabled")
        self.chat_combo.configure(state="readonly" if idle else "disabled")

    def on_message(self, role: str, text: str) -> None:
        if not text.strip():
            return
        prefix = self.role_prefix.get(role, role)
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", f"{prefix}: {text}\n\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def on_history_replaced(self, messages: list[dict]) -> None:
        # Новый или загруженный чат — перерисовываем лог с нуля.
        self.log_widget.configure(state="normal")
        self.log_widget.delete("1.0", "end")
        self.log_widget.configure(state="disabled")
        for m in messages:
            role = m.get("role", "")
            if role in ("user", "assistant"):
                self.on_message(role, m.get("content", ""))

    def on_chat_saved(self, _title: str, path: Path) -> None:
        # Автосохранение после хода диалога — обновляем список и подсвечиваем текущий чат.
        self._refresh_chat_list(select_path=path)

    # ---- обработчики ввода ----

    def _on_press_talk(self, _event=None) -> None:
        # Тот же смысл, что был у зажатия пробела: во время SPEAKING — barge-in,
        # во время IDLE — начать новый ход.
        if self.app.ctx.state is State.SPEAKING:
            self.app.interrupt.set()
        elif self.app.ctx.state is State.IDLE:
            self.app.push_to_talk.set()

    def _on_release_talk(self, _event=None) -> None:
        # Останавливает запись, если она идёт; в остальных состояниях безвредно.
        self.app.interrupt.set()

    def _on_send_text(self, _event=None) -> None:
        text = self.entry_var.get().strip()
        if not text or self.app.ctx.state is not State.IDLE:
            return
        self.entry_var.set("")
        self.app.text_input = text
        self.app.push_to_talk.set()

    def _on_new_chat(self) -> None:
        self.app.new_chat()
        self.chat_combo.set("")

    def _on_chat_selected(self, _event=None) -> None:
        idx = self.chat_combo.current()
        if idx < 0 or idx >= len(self._chat_options):
            return
        _, path = self._chat_options[idx]
        self.app.load_chat(path)

    def _refresh_chat_list(self, select_path: Path | None = None) -> None:
        chats = history.list_chats()
        # Если два чата случайно называются одинаково — различаем по имени файла.
        seen: dict[str, int] = {}
        options: list[tuple[str, Path]] = []
        for chat in chats:
            seen[chat.title] = seen.get(chat.title, 0) + 1
            label = chat.title if seen[chat.title] == 1 else f"{chat.title} ({chat.chat_id})"
            options.append((label, chat.path))

        self._chat_options = options
        self.chat_combo["values"] = [label for label, _ in options]

        if select_path is not None:
            for i, (_, path) in enumerate(options):
                if path == select_path:
                    self.chat_combo.current(i)
                    break

    def _on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.app.stop_all.set()

    # ---- насос событий tkinter внутри asyncio-цикла ----

    async def tick_forever(self, interval: float = 0.03) -> None:
        try:
            while not self._closed:
                self.root.update()
                await asyncio.sleep(interval)
        except tk.TclError:
            # Окно уже уничтожено (например, пользователь закрыл его иначе).
            pass
        finally:
            self._closed = True
            self.app.stop_all.set()

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass