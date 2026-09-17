import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from config import ASRConfig, ASREngine
from logger import log


MODEL_CACHE_DIR: Final = Path(__file__).resolve().parent.parent.parent / ".cache" / "asr"

# аудио
SAMPLE_RATE: Final = 16000          # частота, которую ждёт whisper
CHANNELS: Final = 1
DTYPE: Final = "int16"
BLOCK_MS: Final = 30                # размер блока записи, мс
MAX_RECORD_SECONDS: Final = 90      # страховка, если "отпускание" потерялось

@dataclass
class AudioClip:
    pcm: bytes
    sample_rate: int = SAMPLE_RATE

    def to_float32(self) -> np.ndarray:
        arr = np.frombuffer(self.pcm, dtype=np.int16).astype(np.float32)
        return arr / 32768.0


class FasterWhisperASR:
    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self._model = None

    async def load(self) -> None:
        from faster_whisper import WhisperModel

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        def _load():
            # Сначала пробуем СТРОГО без сети — если модель уже скачана в
            # прошлый раз, huggingface_hub не будет даже проверять версию
            # на сервере, и загрузка станет мгновенной и не требующей
            # интернета. Если модели ещё нет (первый запуск) — упадёт,
            # тогда просто повторяем уже с доступом в сеть.
            try:
                return WhisperModel(
                    self.cfg.asr_model, device=self.cfg.asr_device,
                    compute_type=self.cfg.asr_compute,
                    download_root=str(MODEL_CACHE_DIR), local_files_only=True,
                )
            except Exception:
                log.info("модель ASR не найдена в локальном кеше, скачиваю...")
                return WhisperModel(
                    self.cfg.asr_model, device=self.cfg.asr_device,
                    compute_type=self.cfg.asr_compute,
                    download_root=str(MODEL_CACHE_DIR),
                )

        self._model = await asyncio.to_thread(_load)
        log.info("ASR: faster-whisper (%s, %s/%s) загружен из кеша", self.cfg.asr_model, self.cfg.asr_device, self.cfg.asr_compute)

    async def transcribe(self, audio: AudioClip) -> str:
        if not audio.pcm or len(audio.pcm) < int(SAMPLE_RATE * 0.2) * 2:
            return ""

        def run() -> str:
            segments, _info = self._model.transcribe(
                audio.to_float32(), language=self.cfg.asr_language, vad_filter=True, beam_size=5,
            )
            return "".join(seg.text for seg in segments).strip()

        return await asyncio.to_thread(run)

    async def unload(self) -> None:
        self._model = None


ASR_BACKENDS: dict[ASREngine, type] = {
    ASREngine.FASTER_WHISPER: FasterWhisperASR,
}