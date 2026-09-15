import asyncio

from config import TTSConfig, TTSEngine
from logger import log


class EdgeTTS:
    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    async def load(self) -> None:
        log.info("TTS: edge-tts (%s)", self.cfg.tts_voice)

    async def speak(self, text: str, stop: asyncio.Event) -> None:
        # TODO: edge_tts.Communicate(text, voice).stream() -> декодировать mp3-чанки
        # через av и играть в sd.OutputStream, проверяя stop.is_set() между чанками
        await asyncio.sleep(0.3)

    async def unload(self) -> None:
        pass


TTS_BACKENDS: dict[TTSEngine, type] = {
    TTSEngine.EDGE_TTS: EdgeTTS,
}