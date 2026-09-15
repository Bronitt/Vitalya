import argparse
import asyncio
import signal

import logger
from logger import log
from config import EngineConfig
from engine import Assistant, Engines
from state import Context, State, on_state_change
from ui import AssistantUI


def parse_args():
    parser = argparse.ArgumentParser(description="Agent v1")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


async def main() -> None:
    ctx = Context()
    ctx.listeners.append(on_state_change)
    config = EngineConfig.load_config()
    app = Assistant(Engines(config), ctx)

    ui = AssistantUI(app)
    ctx.listeners.append(ui.on_state_change)
    ctx.message_listeners.append(ui.on_message)

    loop = asyncio.get_running_loop()
    # Ctrl+C (если запущено из терминала) всё ещё гасит корректно:
    # доваривает ход, выгружает модели. Основной способ выйти — закрыть окно.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, app.stop_all.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: app.stop_all.set())  # Windows

    log.info("Окно открыто. Зажми кнопку 'Зажми и говори' или напиши текст. Закрой окно — выход.")

    # ui.tick_forever() крутит tkinter.root.update() на том же потоке,
    # что и event loop, поэтому обработчики окна безопасно трогают
    # asyncio.Event напрямую (см. ui.py).
    ui_task = asyncio.create_task(ui.tick_forever())
    try:
        await app.run()
    finally:
        ui_task.cancel()
        try:
            await ui_task
        except asyncio.CancelledError:
            pass
        ui.destroy()


if __name__ == '__main__':
    args = parse_args()
    logger.setup_logger(args.debug)

    asyncio.run(main())