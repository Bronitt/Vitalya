import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from config import LLMConfig, LLMEngine
from logger import log
from agent.tools import TOOL_SCHEMAS
from main import PROJECT_ROOT_PATH


GEMINI_KEY_URL: Final[str] = "https://aistudio.google.com/apikey"
ENV_PATH: Path = PROJECT_ROOT_PATH / ".env"


# Костыль, который позже следует вынести в отдельный модуль для работы с .env,
# когда будет больше переменных окружения
def _ensure_env_file() -> None:
    """Создаёт .env с пустым GEMINI_API_KEY, если файла ещё нет. Сам .env в
     git не коммитится (см. .gitignore), поэтому у каждого, кто клонирует
     проект, он появится сам при первом запуске вместо ручного копирования
     .env.example."""
    if ENV_PATH.exists():
        return
    ENV_PATH.write_text(
        f"# ключ можно получить бесплатно тут: {GEMINI_KEY_URL}\n"
        "GEMINI_API_KEY=\n",
        encoding="utf-8",
    )
    log.warning(
        "создан пустой файл %s — впиши туда GEMINI_API_KEY (взять можно тут: %s), "
        "если планируешь использовать облачный LLM-бэкенд (llm.llm_engine = \"gemini\")",
        ENV_PATH, GEMINI_KEY_URL,
    )


_ensure_env_file()
load_dotenv(ENV_PATH)
load_dotenv()


TITLE_PROMPT: Final[str] = (
    "Придумай короткий заголовок чата (до 5 слов, без кавычек и точки в конце) "
    "по первому сообщению пользователя. Ответь ТОЛЬКО заголовком, без пояснений.\n\n"
    "Сообщение пользователя: {message}"
)


class OllamaLLM:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = None

    async def load(self) -> None:
        import ollama
        self._client = ollama.AsyncClient(host=self.cfg.llm_host)
        log.info("LLM: ollama (%s @ %s)", self.cfg.llm_model, self.cfg.llm_host)

    async def think(self, history: list[dict]) -> dict:
        resp = await self._client.chat(
            model=self.cfg.llm_model,
            messages=history,
            tools=TOOL_SCHEMAS,
            think=self.cfg.llm_think,
            options={
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict
            },
        )
        msg = resp["message"]
        log.debug("LLM raw: %r", msg)
        calls = msg.get("tool_calls") or []
        tool_call = None
        if calls:
            c = calls[0]["function"]
            tool_call = {"name": c["name"], "arguments": c.get("arguments", {})}
        return {"text": msg.get("content", ""), "tool_call": tool_call}

    async def summarize_title(self, first_message: str) -> str:
        try:
            resp = await self._client.chat(
                model=self.cfg.llm_model,
                messages=[{"role": "user", "content": TITLE_PROMPT.format(message=first_message[:500])}],
                think=False,  # заголовку рассуждения не нужны, только результат
                options={"temperature": 0.3, "num_predict": 30},
            )
            title = resp["message"].get("content", "").strip().strip('"').strip("«»")
            return title[:60] if title else ""
        except Exception:
            log.warning("не удалось сгенерировать заголовок чата через LLM")
            return ""

    async def unload(self) -> None:
        self._client = None


# ---- Gemini API ----

def _schemas_to_gemini(schemas: list[dict]) -> list[dict]:
    """TOOL_SCHEMAS у нас в OpenAI/Ollama-формате ({"type": "function", "function": {...}}),
    с обычным JSON Schema (type: "object"/"string" в нижнем регистре).

    ВАЖНО: используем именно parameters_json_schema, а не parameters —
    поле parameters ожидает types.Schema с type в ВЕРХНЕМ регистре
    ("OBJECT"/"STRING"), и наш нижний регистр там просто не матчится
    (см. https://github.com/googleapis/python-genai/issues/11), из-за чего
    модель получает нерабочее объявление функции и никогда не вызывает
    инструменты. parameters_json_schema принимает обычный JSON Schema
    как есть, без конвертации регистра."""
    declarations = []
    for s in schemas:
        fn = s["function"]
        declarations.append({
            "name": fn["name"],
            "description": fn["description"],
            "parameters_json_schema": fn["parameters"],
        })
    return [{"function_declarations": declarations}]


def _history_to_gemini(history: list[dict]) -> tuple[str | None, list[dict]]:
    """Ollama-стиль истории (role: system/user/assistant/tool) -> формат Gemini
    (system отдельно, роли user/model, contents с parts). Инструменты Gemini
    подтверждает через function_response, а не через отдельную роль "tool" —
    но для простоты (и раз у нас нет параллельных вызовов) кладём результат
    инструмента как обычное текстовое сообщение от пользователя с пометкой:
    этого достаточно, чтобы модель поняла контекст и продолжила диалог."""
    system_text = None
    contents = []
    for m in history:
        role = m.get("role")
        text = m.get("content", "")
        if role == "system":
            system_text = text
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
        elif role == "tool":
            contents.append({"role": "user", "parts": [{"text": f"[Результат инструмента]: {text}"}]})
    return system_text, contents


class GeminiLLM:
    """Ключ берётся из переменной окружения (.env), а не из
    config.toml"""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = None


    async def _verify_model(self) -> None:
        """Проверяем, что llm_model реально существует у Gemini И поддерживает
        generateContent, ДО первого вызова think(). Модель может присутствовать
        в списке client.models.list(), но не поддерживать нужный нам метод
        (например, это embedding-модель или устаревшая версия) — именно так
        получился 404 "is not supported for generateContent" при прежней
        проверке на простое наличие имени."""
        import asyncio

        def list_supporting_generate() -> set[str]:
            names = set()
            for m in self._client.models.list():
                if "generateContent" in (m.supported_actions or []):
                    names.add(m.name.removeprefix("models/"))
            return names

        try:
            available = await asyncio.to_thread(list_supporting_generate)
        except Exception as e:
            log.warning("не удалось проверить список моделей Gemini: %s", e)
            return

        requested = self.cfg.llm_model.removeprefix("models/")
        if requested not in available:
            raise RuntimeError(
                f"модель '{self.cfg.llm_model}' недоступна для generateContent в Gemini API.\n"
                f"Доступные модели:\n{', '.join(sorted(available))}\n"
                f"Проверь llm.llm_model в config.toml"
            )

    async def load(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                f"GEMINI_API_KEY пуст в {ENV_PATH}. "
                f"Получить ключ можно тут: {GEMINI_KEY_URL}"
            )
        from google import genai
        self._client = genai.Client(api_key=api_key)

        await self._verify_model()
        log.info("LLM: gemini (%s)", self.cfg.llm_model)

    async def think(self, history: list[dict]) -> dict:
        import asyncio
        from google.genai import types, errors

        system_text, contents = _history_to_gemini(history)
        tools = _schemas_to_gemini(TOOL_SCHEMAS)

        config_kwargs = {"temperature": self.cfg.temperature, "tools": tools}
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if self.cfg.num_predict != -1:
            config_kwargs["max_output_tokens"] = self.cfg.num_predict

        def call():
            return self._client.models.generate_content(
                model=self.cfg.llm_model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        # 503/500 у Gemini — почти всегда временная перегрузка серверов, а не
        # ошибка запроса, так что имеет смысл пару раз попробовать снова
        # с растущей паузой, прежде чем сдаваться.
        delays = [1, 3, 7]
        last_error: Exception | None = None
        for attempt, delay in enumerate([0] + delays):
            if delay:
                await asyncio.sleep(delay)
            try:
                resp = await asyncio.to_thread(call)
                break
            except errors.ServerError as e:
                last_error = e
                log.warning("Gemini недоступен (попытка %d/%d): %s", attempt + 1, len(delays) + 1, e)
        else:
            log.error("Gemini не ответил после %d попыток: %s", len(delays) + 1, last_error)
            return {
                "text": "Сервер Gemini сейчас перегружен, попробуй ещё раз через минуту.",
                "tool_call": None,
            }

        log.debug("Gemini raw: %r", resp)

        tool_call = None
        text = ""
        candidate = resp.candidates[0] if resp.candidates else None
        if candidate:
            for part in candidate.content.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_call = {"name": fc.name, "arguments": dict(fc.args or {})}
                elif getattr(part, "text", None):
                    text += part.text

        return {"text": text, "tool_call": tool_call}

    async def summarize_title(self, first_message: str) -> str:
        import asyncio
        from google.genai import types, errors

        def call():
            return self._client.models.generate_content(
                model=self.cfg.llm_model,
                contents=[{"role": "user", "parts": [{"text": TITLE_PROMPT.format(message=first_message[:500])}]}],
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=30),
            )

        try:
            resp = await asyncio.to_thread(call)
        except (errors.ServerError, errors.ClientError):
            log.warning("не удалось сгенерировать заголовок чата через Gemini")
            return ""

        text = ""
        candidate = resp.candidates[0] if resp.candidates else None
        if candidate:
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    text += part.text
        title = text.strip().strip('"').strip("«»")
        return title[:60] if title else ""

    async def unload(self) -> None:
        self._client = None


LLM_BACKENDS: dict[LLMEngine, type] = {
    LLMEngine.OLLAMA: OllamaLLM,
    LLMEngine.GEMINI: GeminiLLM,
}