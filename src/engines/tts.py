import asyncio
import io

import av
import sounddevice as sd

from config import TTSConfig, TTSEngine
from logger import log


class EdgeTTS:
    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg

    async def load(self) -> None:
        log.info("TTS: edge-tts (%s)", self.cfg.tts_voice)

    async def speak(self, text: str, stop: asyncio.Event) -> None:
        if not text.strip():
            return

        mp3_bytes = await self._synthesize(text, stop)
        if not mp3_bytes or stop.is_set():
            return

        # Декодирование mp3 и воспроизведение — блокирующие вызовы, уводим в поток,
        # чтобы не морозить event loop. stop проверяем и там (безопасно читать
        # asyncio.Event.is_set() из другого потока — это обычное чтение bool под GIL).
        await asyncio.to_thread(self._play, mp3_bytes, stop)

    async def _synthesize(self, text: str, stop: asyncio.Event) -> bytes:
        import edge_tts
        chunks: list[bytes] = []
        try:
            communicate = edge_tts.Communicate(text, voice=self.cfg.tts_voice)
            async for chunk in communicate.stream():
                if stop.is_set():
                    log.info("озвучка прервана до начала воспроизведения")
                    return b""
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
        except Exception:
            log.exception("не удалось получить аудио от edge-tts")
            return b""
        return b"".join(chunks)

    @staticmethod
    def _play(mp3_bytes: bytes, stop: asyncio.Event) -> None:
        container = av.open(io.BytesIO(mp3_bytes), format="mp3")
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.codec_context.sample_rate

        # Приводим декодированные фреймы к единому int16-моно формату,
        # который понимает sounddevice, не трогая частоту дискретизации.
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

        try:
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as out:
                for frame in container.decode(audio_stream):
                    if stop.is_set():
                        log.info("озвучка прервана (barge-in)")
                        return
                    for resampled in resampler.resample(frame):
                        pcm = resampled.to_ndarray().reshape(-1)
                        out.write(pcm)
                        if stop.is_set():
                            log.info("озвучка прервана (barge-in)")
                            return
        except Exception:
            log.exception("ошибка воспроизведения TTS")
        finally:
            container.close()

    async def unload(self) -> None:
        pass


TTS_BACKENDS: dict[TTSEngine, type] = {
    TTSEngine.EDGE_TTS: EdgeTTS,
}