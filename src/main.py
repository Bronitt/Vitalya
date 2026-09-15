import argparse
import asyncio
import signal

import keyboard

import logger
from logger import log
from config import EngineConfig
from engine import Assistant, Engines
from state import Context, State, on_state_change


def parse_args():
    parser = argparse.ArgumentParser(description="Agent v1")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def wire_key(app: Assistant, loop: asyncio.AbstractEventLoop, config: EngineConfig) -> None:
    def on_press_p2t(_event) -> None:
        # Во время SPEAKING нажатие — это barge-in (прервать реплику), а не новый ход.
        if app.ctx.state is State.SPEAKING:
            loop.call_soon_threadsafe(app.interrupt.set)
        elif app.ctx.state is State.IDLE:
            loop.call_soon_threadsafe(app.push_to_talk.set)

    def on_release_p2t(_event) -> None:
        # Останавливает запись, если она идёт; в остальных состояниях безвредно.
        loop.call_soon_threadsafe(app.interrupt.set)

    keyboard.on_press_key(config.asr.push_to_talk_key_name, on_press_p2t)
    keyboard.on_release_key(config.asr.push_to_talk_key_name, on_release_p2t)

    keyboard.on_press_key(config.core.quit_key, lambda _e: loop.call_soon_threadsafe(app.stop_all.set))


async def main() -> None:
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

    wire_key(app, loop, config)
    log.info("Зажми и держи %s, чтобы говорить. %s — выход.", config.asr.push_to_talk_key_name, config.core.quit_key.upper())
    await app.run()


if __name__ == '__main__':
    args = parse_args()
    logger.setup_logger(args.debug)

    asyncio.run(main())