import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from logger import log


class State(Enum):
    BOOT = "boot"  # загрузка моделей
    IDLE = "idle"  # ждём нажатия клавиши
    LISTENING = "listening"  # пишем звук с микрофона
    THINKING = "thinking"  # ASR + LLM
    ACTING = "acting"  # выполняем инструмент в ОС
    SPEAKING = "speaking"  # играем TTS
    ERROR = "error"  # ошибка, восстанавливаемся
    SHUTDOWN = "shutdown"  # выгружаем ресурсы

TRANSITIONS: dict[State, set[State]] = {
    State.BOOT:      {State.IDLE, State.ERROR, State.SHUTDOWN},
    State.IDLE:      {State.LISTENING, State.THINKING, State.SHUTDOWN},
    State.LISTENING: {State.THINKING, State.IDLE, State.ERROR},
    State.THINKING:  {State.ACTING, State.SPEAKING, State.IDLE, State.ERROR},
    State.ACTING:    {State.THINKING, State.SPEAKING, State.ERROR},
    State.SPEAKING:  {State.IDLE, State.ERROR},
    State.ERROR:     {State.IDLE, State.SHUTDOWN},
    State.SHUTDOWN:  set(),
}


@dataclass
class Context:
    #Всё изменяемое состояние живёт здесь, а не в глобалках.
    state: State = State.BOOT
    history: list[dict] = field(default_factory=list)
    transcript: str = ""
    reply: str = ""
    pending_tool: dict | None = None
    error_count: int = 0
    sandbox: Path = field(default_factory=lambda: Path("./sandbox").resolve())
    # приложения, запущенные агентом через open_app: имя -> Popen
    # печатать можно только сюда — не в произвольные окна на рабочем столе
    running_apps: dict[str, subprocess.Popen] = field(default_factory=dict)

    # подписчики на смену состояния (аватар, UI, логи)
    listeners: list[Callable[[State, State], None]] = field(default_factory=list)
    # подписчики на новые реплики (роль "user"/"assistant" + текст) — для UI-лога
    message_listeners: list[Callable[[str, str], None]] = field(default_factory=list)
    # подписчики на полную замену истории (новый/загруженный чат) — UI перерисовывает лог
    history_listeners: list[Callable[[list[dict]], None]] = field(default_factory=list)

    def set_state(self, new: State) -> None:
        if new not in TRANSITIONS[self.state]:
            raise RuntimeError(f"Недопустимый переход: {self.state.value} -> {new.value}")
        old, self.state = self.state, new
        log.info("состояние: %s -> %s", old.value, new.value)
        for fn in self.listeners:
            try:
                fn(old, new)
            except Exception as e:
                log.warning("слушатель состояния упал: %s", e)

    def emit_message(self, role: str, text: str) -> None:
        for fn in self.message_listeners:
            try:
                fn(role, text)
            except Exception as e:
                log.warning("слушатель сообщений упал: %s", e)

    def emit_history_replaced(self) -> None:
        # Системное сообщение UI не касается — отдаём только видимую часть диалога.
        visible = [m for m in self.history if m.get("role") != "system"]
        for fn in self.history_listeners:
            try:
                fn(visible)
            except Exception as e:
                log.warning("слушатель истории упал: %s", e)


def on_state_change(old: State, new: State) -> None:
    # Сюда подключается аватар: одна анимация на состояние.
    avatar = {
        State.IDLE: "idle",
        State.LISTENING: "listening",
        State.THINKING: "thinking",
        State.ACTING: "working",
        State.SPEAKING: "talking",
        State.ERROR: "confused",
    }.get(new)
    if avatar:
        pass  # TODO: переключить кадр/анимацию Live2D