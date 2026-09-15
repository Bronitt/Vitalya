import argparse
import asyncio
import signal

import keyboard

import logger
from logger import log
from config import EngineConfig
from engine import Assistant, Engines
from state import Context, State, on_state_change

PUSH_TO_TALK_KEY = "space"


def parse_args():
    parser = argparse.ArgumentParser(description="Agent v1")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--demo", action="store_true", help="Один автопрогон вместо push-to-talk (без клавиатуры/микрофона)")
    return parser.parse_args()


def wire_push_to_talk(app: Assistant, loop: asyncio.AbstractEventLoop, push2talk_button:str = "space") -> None:
    def on_press(_event) -> None:
        # Во время SPEAKING нажатие — это barge-in (прервать реплику), а не новый ход.
        if app.ctx.state is State.SPEAKING:
            loop.call_soon_threadsafe(app.interrupt.set)
        elif app.ctx.state is State.IDLE:
            loop.call_soon_threadsafe(app.push_to_talk.set)

    def on_release(_event) -> None:
        # Останавливает запись, если она идёт; в остальных состояниях безвредно.
        loop.call_soon_threadsafe(app.interrupt.set)

    keyboard.on_press_key(push2talk_button, on_press)
    keyboard.on_release_key(push2talk_button, on_release)


async def main(demo: bool) -> None:
    ctx = Context()
    ctx.listeners.append(on_state_change)
    config = EngineConfig.load_config()
    app = Assistant(Engines(config), ctx)

    loop = asyncio.get_running_loop()
    # Ctrl+C гасит корректно: доваривает ход, выгружает модели.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, app.stop_all.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: app.stop_all.set())  # Windows

    if demo:
        async def demo_trigger():
            await asyncio.sleep(0.2)
            app.push_to_talk.set()
            await asyncio.sleep(2.0)
            app.interrupt.set()  # имитируем отпускание клавиши
            await asyncio.sleep(0.5)
            app.stop_all.set()
        await asyncio.gather(app.run(), demo_trigger())
    else:
        wire_push_to_talk(app, loop, config.asr.push_to_talk_button_name)
        log.info("Зажми и держи %s, чтобы говорить. Ctrl+C — выход.", config.asr.push_to_talk_button_name)
        await app.run()


if __name__ == '__main__':
    args = parse_args()
    logger.setup_logger(args.debug)

    asyncio.run(main(args.demo))