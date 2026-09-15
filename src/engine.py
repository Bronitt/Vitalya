import asyncio
from pathlib import Path
from typing import Callable, Final, Protocol

import numpy as np
import sounddevice as sd

import history as chat_history
from config import EngineConfig
from engines.llm import LLM_BACKENDS
from logger import log
from state import State, Context
from tools import TOOL_REGISTRY
from engines.transcriber import AudioClip, ASR_BACKENDS, SAMPLE_RATE, BLOCK_MS, CHANNELS, DTYPE, MAX_RECORD_SECONDS
from engines.tts import TTS_BACKENDS


# ---- контракты бэкендов ----
# Любой новый движок должен реализовать этот протокол и зарегистрироваться
# в соответствующем словаре ниже — переключение делается одним полем в EngineConfig.

class ASRBackend(Protocol):
    async def load(self) -> None: ...
    async def transcribe(self, audio: AudioClip) -> str: ...
    async def unload(self) -> None: ...


class LLMBackend(Protocol):
    async def load(self) -> None: ...
    async def think(self, history: list[dict]) -> dict: ...
    async def unload(self) -> None: ...


class TTSBackend(Protocol):
    async def load(self) -> None: ...
    async def speak(self, text: str, stop: asyncio.Event) -> None: ...
    async def unload(self) -> None: ...


class Engines:
    """Единая точка доступа к ASR/LLM/TTS. Конкретные движки выбираются
        в EngineConfig — Assistant и остальной код об этом не знает."""

    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()
        self._asr: ASRBackend = ASR_BACKENDS[self.cfg.asr.asr_engine](self.cfg.asr)
        self._llm: LLMBackend = LLM_BACKENDS[self.cfg.llm.llm_engine](self.cfg.llm)
        self._tts: TTSBackend = TTS_BACKENDS[self.cfg.tts.tts_engine](self.cfg.tts)


    async def load(self) -> None:
        log.info("загрузка моделей...")
        await self._asr.load()
        await self._llm.load()
        await self._tts.load()


    async def record(self, stop: asyncio.Event) -> AudioClip:
        frames: list[np.ndarray] = []

        def callback(indata, frame_count, time_info, status):
            if status:
                log.warning("аудио-статус: %s", status)
            frames.append(indata.copy())

        blocksize = int(SAMPLE_RATE * BLOCK_MS / 1000)
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                            blocksize=blocksize, callback=callback):
            try:
                await asyncio.wait_for(stop.wait(), timeout=MAX_RECORD_SECONDS)
            except asyncio.TimeoutError:
                log.warning("запись остановлена по таймауту (%d с)", MAX_RECORD_SECONDS)

        if not frames:
            return AudioClip(pcm=b"")
        audio = np.concatenate(frames, axis=0).reshape(-1)
        return AudioClip(pcm=audio.tobytes())


    async def transcribe(self, audio: AudioClip) -> str:
        return await self._asr.transcribe(audio)


    async def think(self, history: list[dict]) -> dict:
        return await self._llm.think(history)


    async def speak(self, text: str, stop: asyncio.Event) -> None:
        await self._tts.speak(text, stop)


    async def unload(self) -> None:
        await self._asr.unload()
        await self._llm.unload()
        await self._tts.unload()
        log.info("выгрузка моделей")


MAX_ERRORS: Final = 3


class Assistant:
    def __init__(self, engines: Engines, ctx: Context) -> None:
        self.engines = engines
        self.ctx = ctx
        self.stop_all = asyncio.Event()  # глобальное завершение (закрытие окна / Ctrl+C)
        self.interrupt = asyncio.Event()  # прерывание текущей реплики / записи
        self.push_to_talk = asyncio.Event()
        # Если не None — turn() возьмёт этот текст вместо записи+ASR.
        # Выставляется UI при отправке текста из поля ввода.
        self.text_input: str | None = None

        # --- текущий чат ---
        # None = ещё не сохранённый новый чат; файл появится после первого хода.
        self.chat_path: Path | None = None
        self.chat_title: str = chat_history.DEFAULT_TITLE
        # UI подписывается сюда, чтобы обновить список чатов после автосохранения.
        self.on_chat_saved: Callable[[str, Path], None] | None = None

    # --- управление чатами --------------------------------------------------

    def _system_message(self) -> dict | None:
        char = self.engines.cfg.char
        text = char.char_prompt.replace("{name}", char.char_name).strip()
        return {"role": "system", "content": text} if text else None

    def new_chat(self) -> None:
        """Начинает пустой чат (кнопка "Новый чат" в UI). Не мешает середине хода диалога."""
        if self.ctx.state is not State.IDLE:
            return
        c = self.ctx
        self.chat_path = None
        self.chat_title = chat_history.DEFAULT_TITLE
        c.history = []
        sys_msg = self._system_message()
        if sys_msg:
            c.history.append(sys_msg)
        c.transcript = ""
        c.reply = ""
        c.emit_history_replaced()

    def load_chat(self, path: Path) -> None:
        """Загружает сохранённый чат с диска и подставляет в контекст для LLM."""
        if self.ctx.state is not State.IDLE:
            return
        try:
            title, messages = chat_history.load_chat(path)
        except (OSError, ValueError):
            log.exception("не удалось загрузить чат %s", path)
            return
        c = self.ctx
        self.chat_path = Path(path)
        self.chat_title = title
        c.history = []
        sys_msg = self._system_message()
        if sys_msg:
            c.history.append(sys_msg)
        c.history.extend(messages)
        c.transcript = ""
        c.reply = ""
        c.emit_history_replaced()

    def _save_chat(self) -> None:
        # Системный промпт не сохраняем — он пересобирается заново из char-конфига
        # при загрузке, чтобы старые чаты подхватывали актуальный характер/имя.
        messages = [m for m in self.ctx.history if m.get("role") != "system"]
        if not messages:
            return
        if self.chat_path is None:
            first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            self.chat_title = chat_history.derive_title(first_user)
            self.chat_path = chat_history.unique_path(self.chat_title)
        try:
            chat_history.save_chat(self.chat_path, self.chat_title, messages)
        except OSError:
            log.exception("не удалось сохранить чат %s", self.chat_path)
            return
        if self.on_chat_saved:
            try:
                self.on_chat_saved(self.chat_title, self.chat_path)
            except Exception:
                log.exception("колбэк сохранения чата упал")

    # --- один ход диалога -------------------------------------------------

    async def turn(self) -> None:
        c = self.ctx

        if self.text_input is not None:
            # Текст пришёл из UI напрямую — записи и ASR не требуется.
            c.transcript = self.text_input
            self.text_input = None
        else:
            c.set_state(State.LISTENING)
            audio = await self.engines.record(self.interrupt)
            c.transcript = await self.engines.transcribe(audio)

        if not c.transcript.strip():
            log.info("тишина, возвращаемся в ожидание")
            c.set_state(State.IDLE)
            return

        # Сначала пишем в лог, что распознали, и только потом переключаем
        # состояние — иначе "состояние: -> thinking" печаталось раньше текста.
        log.info("распознано: %s", c.transcript)
        c.set_state(State.THINKING)
        c.history.append({"role": "user", "content": c.transcript})
        c.emit_message("user", c.transcript)

        # Цикл "модель -> инструмент -> модель". Лимит защищает от зацикливания.
        for _ in range(3):
            out = await self.engines.think(c.history)
            if not out.get("tool_call"):
                c.reply = out["text"]
                break

            c.set_state(State.ACTING)
            result = await self.run_tool(out["tool_call"])
            c.history.append({"role": "tool", "content": result})
            c.set_state(State.THINKING)
        else:
            c.reply = "Слишком много шагов, останавливаюсь."

        c.history.append({"role": "assistant", "content": c.reply})
        c.emit_message("assistant", c.reply)
        self._save_chat()
        c.set_state(State.SPEAKING)
        # interrupt мог остаться взведённым с момента отпускания клавиши записи —
        # очищаем перед озвучкой, чтобы TTS получила чистый сигнал прерывания.
        self.interrupt.clear()
        await self.engines.speak(c.reply, self.interrupt)
        c.set_state(State.IDLE)

    async def run_tool(self, call: dict) -> str:
        name, args = call["name"], call.get("arguments", {})
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return f"Инструмент {name} не найден"
        try:
            # Инструменты синхронные -> в отдельный поток, чтобы не блокировать цикл.
            return await asyncio.to_thread(fn, self.ctx, **args)
        except PermissionError as e:
            log.warning("блокировка песочницы: %s", e)
            return f"Отказано: {e}"
        except Exception as e:
            log.exception("инструмент %s упал", name)
            return f"Ошибка инструмента: {e}"

    # --- главный цикл -----------------------------------------------------

    async def run(self) -> None:
        c = self.ctx
        try:
            await self.engines.load()
            c.sandbox.mkdir(parents=True, exist_ok=True)

            # Системный промпт подсаживаем в историю один раз при старте —
            # дальше он просто едет первым сообщением во всех вызовах LLM.
            if not c.history:
                sys_msg = self._system_message()
                if sys_msg:
                    c.history.append(sys_msg)

            c.set_state(State.IDLE)
        except Exception:
            log.exception("не удалось загрузиться")
            c.set_state(State.ERROR)
            return

        while not self.stop_all.is_set():
            try:
                # Ждём либо нажатие кнопки/отправку текста, либо сигнал выхода.
                await self.wait_any(self.push_to_talk, self.stop_all)
                if self.stop_all.is_set():
                    break
                self.push_to_talk.clear()
                self.interrupt.clear()

                await self.turn()
                c.error_count = 0

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ошибка в ходе диалога")
                c.error_count += 1
                # ERROR достижим не из всех состояний, поэтому переводим аккуратно.
                if c.state not in (State.ERROR, State.SHUTDOWN):
                    try:
                        c.set_state(State.ERROR)
                    except RuntimeError:
                        c.state = State.ERROR
                if c.error_count >= MAX_ERRORS:
                    log.error("подряд %d ошибок, выключаюсь", c.error_count)
                    break
                c.set_state(State.IDLE)

        await self.shutdown()

    @staticmethod
    async def wait_any(*events: asyncio.Event, poll: float = 0.2) -> None:
        while not any(e.is_set() for e in events):
            await asyncio.sleep(poll)

    async def shutdown(self) -> None:
        c = self.ctx
        if c.state is not State.SHUTDOWN:
            c.state = State.SHUTDOWN
            log.info("состояние: -> shutdown")
        await self.engines.unload()
        log.info("завершено, реплик в истории: %d", len(c.history))