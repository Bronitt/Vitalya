from config import LLMConfig, LLMEngine
from logger import log


class OllamaLLM:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = None

    async def load(self) -> None:
        import ollama
        self._client = ollama.AsyncClient(host=self.cfg.llm_host)
        log.info("LLM: ollama (%s @ %s)", self.cfg.llm_model, self.cfg.llm_host)

    async def think(self, history: list[dict]) -> dict:
        resp = await self._client.chat(model=self.cfg.llm_model, messages=history, tools=TOOL_SCHEMAS)
        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        tool_call = None
        if calls:
            c = calls[0]["function"]
            tool_call = {"name": c["name"], "arguments": c.get("arguments", {})}
        return {"text": msg.get("content", ""), "tool_call": tool_call}

    async def unload(self) -> None:
        self._client = None


LLM_BACKENDS: dict[LLMEngine, type] = {
    LLMEngine.OLLAMA: OllamaLLM,
}