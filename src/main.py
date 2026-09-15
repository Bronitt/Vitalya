import argparse
import asyncio
import signal

import logger
from config import EngineConfig
from engine import Assistant, Engines
from state import Context, on_state_change


# заимствовано из https://github.com/inni918/warashi
def parse_args():
    parser = argparse.ArgumentParser(description="Agent v1")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

async def main() -> None:
    ctx = Context()
    ctx.listeners.append(on_state_change)
    config = EngineConfig.load_config()
    app = Assistant(Engines(config), ctx)

    loop = asyncio.get_event_loop()
    # Ctrl+C гасит корректно: домовариваем ход, выгружаем модели.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, app.stop_all.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: app.stop_all.set())  # Windows

    # Демо-триггер вместо клавиатуры: дёргаем один ход и выходим.
    # TODO: keyboard.on_press_key("space", ...) -> app.push_to_talk.set()
    async def demo():
        await asyncio.sleep(0.2)
        app.push_to_talk.set()
        await asyncio.sleep(2.0)
        app.stop_all.set()

    await asyncio.gather(app.run(), demo())


if __name__ == '__main__':
    args = parse_args()
    logger.setup_logger(args.debug)

    asyncio.run(main())

