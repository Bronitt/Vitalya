import asyncio
from typing import Final, Protocol

import numpy as np
import sounddevice as sd

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
        self.stop_all = asyncio.Event()  # глобальное завершение (Ctrl+C)
        self.interrupt = asyncio.Event()  # прерывание текущей реплики
        self.push_to_talk = asyncio.Event()

    # --- один ход диалога -------------------------------------------------

    async def turn(self) -> None:
        c = self.ctx

        c.set_state(State.LISTENING)
        audio = await self.engines.record(self.interrupt)

        c.set_state(State.THINKING)
        c.transcript = await self.engines.transcribe(audio)
        if not c.transcript.strip():
            log.info("тишина, возвращаемся в ожидание")
            c.set_state(State.IDLE)
            return
        log.info("распознано: %s", c.transcript)
        c.history.append({"role": "user", "content": c.transcript})

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
            c.set_state(State.IDLE)
        except Exception:
            log.exception("не удалось загрузиться")
            c.set_state(State.ERROR)
            return

        while not self.stop_all.is_set():
            try:
                # Ждём либо нажатие клавиши, либо сигнал выхода.
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