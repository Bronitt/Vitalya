from __future__ import annotations

import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from config import DockerConfig
from logger import log


class DockerUnavailable(RuntimeError):
    """Docker недоступен или контейнер не удалось поднять/выполнить в нём команду."""


@dataclass
class ShellResult:
    return_code: int
    stdout: str
    stderr: str


def _is_alive(name: str) -> bool:
    try:
        res = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return res.returncode == 0 and res.stdout.strip() == "true"


class DockerSandbox:
    """Изолированное окружение агента для выполнения shell-команд.

    Жизненный цикл — "по необходимости": контейнер НЕ поднимается при
    старте программы, а лениво создаётся при первом обращении инструмента
    (run_shell). После этого он живёт до закрытия приложения (stop()
    вызывается из Assistant.shutdown()), а не пересоздаётся на каждую
    команду — так между вызовами одного чата сохраняется состояние
    (установленные пакеты, файлы вне /workspace, переменные окружения
    процесса контейнера и т.п.), и не тратится время на повторный запуск.

    Контейнер монтирует ту же песочницу (ctx.sandbox), что видят
    инструменты list_files/create_note — так агент работает с одними и
    теми же файлами что "снаружи", что "внутри" контейнера.
    """

    def __init__(self, cfg: DockerConfig, sandbox_dir: Path) -> None:
        self.cfg = cfg
        self.sandbox_dir = sandbox_dir
        self._container_name: str | None = None
        # Инструменты выполняются в отдельных потоках (asyncio.to_thread) —
        # без лока два параллельных tool-вызова могли бы поднять два контейнера.
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._container_name is not None

    @staticmethod
    def _docker_available() -> bool:
        return shutil.which("docker") is not None

    def ensure_running(self) -> str:
        """Возвращает имя контейнера, поднимая его при необходимости.

        Потокобезопасно: конкурентные вызовы дождутся одного и того же
        контейнера, а не создадут по своему."""
        if not self.cfg.docker_enabled:
            raise DockerUnavailable("Docker-песочница отключена в конфиге")
        if not self._docker_available():
            raise DockerUnavailable("исполняемый файл docker не найден в PATH")

        with self._lock:
            if self._container_name is not None:
                if _is_alive(self._container_name):
                    return self._container_name
                log.warning("контейнер-песочница %s не отвечает, пересоздаю", self._container_name)
                self._container_name = None

            self.sandbox_dir.mkdir(parents=True, exist_ok=True)
            name = f"{self.cfg.docker_container_prefix}-{uuid.uuid4().hex[:8]}"

            cmd = [
                "docker", "run", "-d", "--rm",
                "--name", name,
                "-v", f"{self.sandbox_dir}:/workspace",
                "-w", "/workspace",
                "--memory", self.cfg.docker_mem_limit,
                "--cpus", str(self.cfg.docker_cpus),
                "--pids-limit", "256",
                "--network", self.cfg.docker_network,
                "--security-opt", "no-new-privileges",
                self.cfg.docker_image,
                "sleep", "infinity",
            ]
            log.info("поднимаю docker-песочницу: %s (образ %s)", name, self.cfg.docker_image)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            except subprocess.CalledProcessError as e:
                raise DockerUnavailable(f"не удалось запустить контейнер: {e.stderr.strip()}") from e
            except subprocess.TimeoutExpired as e:
                raise DockerUnavailable("превышен таймаут запуска контейнера (образ ещё качается?)") from e

            self._container_name = name
            return name

    def exec(self, command: str, timeout: int | None = None) -> ShellResult:
        """Выполняет команду (sh -c) внутри контейнера, поднимая его при необходимости."""
        name = self.ensure_running()
        timeout = timeout or self.cfg.docker_command_timeout
        try:
            res = subprocess.run(
                ["docker", "exec", name, "sh", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ShellResult(return_code=-1, stdout="", stderr=f"команда не уложилась в {timeout} с")
        except (subprocess.SubprocessError, OSError) as e:
            return ShellResult(return_code=-1, stdout="", stderr=f"ошибка вызова docker exec: {e}")
        return ShellResult(return_code=res.returncode, stdout=res.stdout, stderr=res.stderr)

    def stop(self) -> None:
        """Останавливает и удаляет контейнер (если он был поднят). Безопасно
        вызывать в любом случае, в том числе если контейнер ни разу не создавался."""
        with self._lock:
            if self._container_name is None:
                return
            log.info("останавливаю docker-песочницу: %s", self._container_name)
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self._container_name],
                    capture_output=True, text=True, timeout=15,
                )
            except (subprocess.SubprocessError, OSError) as e:
                log.warning("не удалось остановить контейнер %s: %s", self._container_name, e)
            self._container_name = None