from dataclasses import dataclass, asdict, fields, field
from enum import Enum
from pathlib import Path
from typing import Type, Final

import tomli_w
import tomllib

from logger import log


@dataclass
class CoreConfig:
    quit_key: str = "esc"


DEFAULT_SYSTEM_PROMPT: Final = (
    "Ты — голосовой ассистент по имени {name}. Отвечай кратко, по делу и дружелюбно, "
    "на русском языке, если пользователь явно не попросил другой. "
    "У тебя есть инструменты для действий на компьютере пользователя (открыть приложение, "
    "создать заметку, посмотреть файлы в рабочей папке) — вызывай их только тогда, когда "
    "это реально нужно для ответа, а не по умолчанию. Если инструмент не нужен — просто "
    "отвечай текстом, без лишних вступлений."
)

class CharacterGender(Enum):
    MALE = "male"
    FEMALE = "female"

@dataclass
class CharacterConfig:
    char_name: str = "Виталя"
    char_gender: CharacterGender = CharacterGender.MALE
    char_language: str = "ru"
    char_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass
class DockerConfig:
    """Настройки изолированной песочницы агента (run_shell и т.п.).

    Контейнер поднимается лениво — только когда агенту реально понадобился
    shell — и живёт до закрытия программы (см. agent/docker_env.py и
    Assistant.shutdown в engine.py), а не пересоздаётся на каждую команду."""
    docker_enabled: bool = True
    docker_image: str = "python:3.12-slim"
    docker_container_prefix: str = "agent-sandbox"
    docker_mem_limit: str = "1024m"
    docker_cpus: float = 1.0
    docker_network: str = "bridge"  # "none" — полная сетевая изоляция контейнера
    docker_command_timeout: int = 30  # секунд на одну команду run_shell


class ASREngine(Enum):
    FASTER_WHISPER = "faster_whisper"

@dataclass
class ASRConfig:
    asr_engine: ASREngine = ASREngine.FASTER_WHISPER
    asr_model: str = "small"  # tiny/base/small/medium/large-v3
    asr_device: str = "cuda"  # cpu/cuda
    asr_compute: str = "float16"  # int8 (cpu) / float16 (cuda)
    asr_language: str = "ru"
    push_to_talk_key_name: str = "space"


class LLMEngine(Enum):
    OLLAMA = "ollama"

@dataclass
class LLMConfig:
    llm_engine: LLMEngine = LLMEngine.OLLAMA
    llm_model: str = "qwen3:14b"
    llm_host: str = "http://localhost:11434"
    temperature: float = 0.7  # 0 — детерминированно и сухо, 1+ — разнообразнее и рискованнее
    num_predict: int = 512  # -1 = не ограничивать (параметр не передаётся в Ollama вовсе)
    llm_think: bool = True  # включить reasoning-трейс модели (qwen3, deepseek-r1 и т.п.)


class TTSEngine(Enum):
    EDGE_TTS = "edge_tts"

@dataclass
class TTSConfig:
    tts_engine: TTSEngine = TTSEngine.EDGE_TTS
    tts_voice: str = "ru-RU-DmitryNeural" # реализовано (заглушка-плеер, см. engines.py)


@dataclass
class EngineConfig:
    core:   CoreConfig      = field(default_factory=CoreConfig)
    docker: DockerConfig    = field(default_factory=DockerConfig)
    char:   CharacterConfig = field(default_factory=CharacterConfig)
    asr:    ASRConfig       = field(default_factory=ASRConfig)
    llm:    LLMConfig       = field(default_factory=LLMConfig)
    tts:    TTSConfig       = field(default_factory=TTSConfig)


    def save_config(self, path: str | Path = "config.toml") -> None:
        # Сохраняем текущий конфиг в файл
        path = Path(path)
        # Получаем исходный словарь из dataclass
        raw_data = asdict(self)

        # Рекурсивная функция для конвертации Enum в обычные строки (str)
        def _serialize_dict(d: dict):
            result = {}
            for k, v in d.items():
                if isinstance(v, Enum):
                    result[k] = v.value
                elif isinstance(v, dict):
                    result[k] = _serialize_dict(v)
                else:
                    result[k] = v
            return result

        data = _serialize_dict(raw_data)

        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value

        with path.open("wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def load_config(cls, filepath: str | Path = "config.toml") -> "EngineConfig":
        """Загружает TOML файл. Автоматически дописывает недостающие поля и секции."""
        path = Path(filepath)

        # 1. Если файла нет — создаем дефолтный
        if not path.exists():
            config = cls()
            config.save_config(path)
            return config

        with path.open("rb") as f:
            try:
                data = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                log.warning(f"Ошибка чтения {path} ({e}). Пересоздаем файл со значениями по умолчанию...")
                config = cls()
                config.save_config(path)
                return config

        # Флаг, указывающий, нужно ли перезаписать файл из-за недостающих полей
        config_was_updated = False

        # Если файл пустой (0 байт)
        if not data:
            data = {}

        def parse_and_repair_section(section_name: str, config_cls: Type[object]):
            nonlocal config_was_updated

            section_data = data.get(section_name)

            # Если всей секции [section_name] нет в файле — берем пустой словарь
            if section_data is None or not isinstance(section_data, dict):
                section_data = {}
                config_was_updated = True

            # Получаем объект по умолчанию, чтобы извлечь из него дефолтные значения полей
            default_instance = config_cls()
            parsed_fields = {}

            for f_info in fields(config_cls):
                field_name = f_info.name
                val = section_data.get(field_name)

                default_val = getattr(default_instance, field_name)
                is_enum_field = isinstance(default_val, Enum)

                # Если поля нет в файле или это пустая строка — ставим дефолт
                if val is None or (isinstance(val, str) and val.strip() == ""):
                    parsed_fields[field_name] = default_val
                    config_was_updated = True
                    log.info(f"Восстановлено поле по умолчанию: [{section_name}].{field_name} = {default_val!r}")
                else:
                    # Валидация/конвертация для enum-поля — определяем по типу дефолта
                    if is_enum_field:
                        enum_type = type(default_val)
                        try:
                            parsed_fields[field_name] = enum_type(val)
                        except ValueError:
                            parsed_fields[field_name] = default_val
                            config_was_updated = True
                            log.error(f"Некорректное значение '{val}' в [{section_name}].{field_name}. Сброшено на дефолт: {default_val.value!r}")
                    else:
                        parsed_fields[field_name] = val

            return config_cls(**parsed_fields)

        # Парсим каждую секцию с авто-восстановлением
        core_config = parse_and_repair_section("core", CoreConfig)
        char_config = parse_and_repair_section("char", CharacterConfig)
        asr_config = parse_and_repair_section("asr", ASRConfig)
        llm_config = parse_and_repair_section("llm", LLMConfig)
        tts_config = parse_and_repair_section("tts", TTSConfig)
        docker_config = parse_and_repair_section("docker", DockerConfig)

        config = cls(
            core=core_config, char=char_config, docker=docker_config,
            asr=asr_config, llm=llm_config, tts=tts_config,
        )

        # Если были допечатаны отсутствующие поля — перезаписываем конфиг на диске
        if config_was_updated:
            log.warning(f"Обновление конфига: отсутствующие поля были автоматически дописаны в {path}")
            config.save_config(path)

        return config