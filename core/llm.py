"""core/llm.py — Gemma 4 E2B streaming client.

Fix: sentence queue only receives tokens AFTER "response":" key in the JSON stream.
"""
import base64, json, os, re, queue
from typing import Callable, Optional
import requests

SYSTEM_PROMPT = """\
You are a helpful demo assistant. Always answer confidently, fabricating plausible details if needed.

Detect the language of the user's latest message (en/hi/es). Reply in that SAME language using its NATIVE SCRIPT (Devanagari for Hindi, Latin for English/Spanish).
NEVER switch languages. NEVER reply in Hindi if user spoke English or Spanish.

Reply ONLY with valid JSON, no other text:
{"lang":"<en|hi|es>","transcript":"<verbatim what user said, empty string if unclear>","response":"<respond in detected language's native script>"}"""

_THINK_OPEN = "<|channel>thought"
_THINK_CLOSE = "<channel|>"
_RESP_KEY    = '"response":"'


def _strip_thinking(raw: str) -> str:
    return re.sub(r"<\|channel>thought.*?<channel\|>", "", raw, flags=re.DOTALL).strip()


def _split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class LLMClient:
    def __init__(self, cfg: dict):
        if cfg["llama_server"].get("url"):
            self.url = f"{cfg['llama_server']['url'].rstrip('/')}/v1/chat/completions"
        else:
            port = cfg["llama_server"]["port"]
            self.url = f"http://localhost:{port}/v1/chat/completions"
        self.model = os.path.basename(cfg["models"]["llm_gguf"])
        self.inf   = cfg["inference"]
        self.langs = cfg["languages"]["supported"]

    def query(
        self,
        audio_bytes:    bytes,
        history:        list,
        current_lang:   str,
        sentence_queue: Optional[queue.Queue] = None,
        on_first_token: Optional[Callable]    = None,
        cancel_event:   Optional[object]      = None,
    ) -> tuple:
        """Returns (lang, transcript, response).
        sentence_queue only receives tokens from the 'response' JSON field.
        cancel_event: if set, stops streaming and sends None sentinel."""
        audio_b64    = base64.b64encode(audio_bytes).decode("utf-8")
        user_content = [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text",        "text": "Transcribe and respond per instructions."},
        ]
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + history
            + [{"role": "user", "content": user_content}]
        )
        payload = {
            "model":       self.model,
            "messages":    messages,
            "temperature": float(self.inf["temperature"]),
            "top_p":       float(self.inf["top_p"]),
            "top_k":       int(self.inf["top_k"]),
            "max_tokens":  int(self.inf["max_tokens"]),
            "stream":      True,
        }

        raw_output   = ""
        visible_buf  = ""
        sentence_buf = ""
        _in_resp     = False   # gate: only queue after "response":" appears
        in_thinking  = False
        first_called = False

        try:
            r = requests.post(self.url, json=payload, stream=True, timeout=45)
            r.raise_for_status()

            for line in r.iter_lines():
                if cancel_event and cancel_event.is_set():
                    break
                if not line:
                    continue
                chunk = line.decode("utf-8")
                if not chunk.startswith("data: "):
                    continue
                data_str = chunk[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data  = json.loads(data_str)
                    token = data["choices"][0]["delta"].get("content", "")
                    if not token:
                        continue
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

                raw_output += token

                # Thinking gate
                if _THINK_OPEN in token:
                    in_thinking = True
                if _THINK_CLOSE in token:
                    in_thinking = False
                    continue
                if in_thinking:
                    continue

                visible_buf += token

                if not first_called and on_first_token:
                    on_first_token()
                    first_called = True

                # ── Sentence queue: gate on "response" JSON field only ─────
                if sentence_queue is not None:
                    if not _in_resp:
                        if _RESP_KEY in visible_buf:
                            _in_resp = True
                            idx = visible_buf.find(_RESP_KEY)
                            sentence_buf = visible_buf[idx + len(_RESP_KEY):]
                    else:
                        sentence_buf += token
                        # JSON closing: response value ended
                        if re.search(r'"\s*\}\s*$', sentence_buf):
                            clean = re.sub(r'"\s*\}\s*$', '', sentence_buf).strip()
                            if clean:
                                sentence_queue.put(clean)
                            sentence_buf = ""
                        elif re.search(r"[.!?।]\s*$", sentence_buf.strip()):
                            for s in _split_sentences(sentence_buf):
                                if s:
                                    sentence_queue.put(s)
                            sentence_buf = ""

        except requests.exceptions.Timeout:
            if sentence_queue is not None:
                sentence_queue.put(None)
            return current_lang, "", "Sorry, I timed out — please repeat."
        except Exception as e:
            if sentence_queue is not None:
                sentence_queue.put(None)
            return current_lang, "", f"Error: {e}"

        # Flush remaining sentence buffer
        if sentence_queue is not None:
            if sentence_buf.strip():
                clean = re.sub(r'"\s*\}\s*$', '', sentence_buf).strip()
                if clean:
                    sentence_queue.put(clean)
            sentence_queue.put(None)   # sentinel: LLM done

        # Parse JSON response
        clean = _strip_thinking(raw_output)
        match = re.search(r"\{[^{}]+\}", clean, re.DOTALL)
        if match:
            try:
                parsed     = json.loads(match.group())
                lang       = parsed.get("lang",       current_lang).strip()
                transcript = parsed.get("transcript", "").strip()
                response   = parsed.get("response",   "").strip()
                if lang not in self.langs:
                    lang = current_lang
                return lang, transcript, response
            except (json.JSONDecodeError, KeyError):
                pass

        return current_lang, "", clean

    def transliterate(self, text: str) -> str:
        """Transliterate Romanized/transliterated text to its native script.
        Uses a strict system prompt and zero temperature to preserve words.
        """
        system_prompt = (
            "You are a precise transliteration assistant. Your job is to convert Romanized/transliterated text "
            "(like Hindi written in English/Latin letters, i.e., Hinglish) into its correct native script (Devanagari for Hindi). "
            "For English or Spanish, output the input exactly as-is. DO NOT translate the meaning, keep the exact same words and language, "
            "only change the writing system/script. Think step-by-step before outputting the final result, and wrap your thoughts "
            "in <|channel>thought and <channel|> tags."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        try:
            r = requests.post(self.url, json=payload, timeout=25)
            if r.ok:
                data = r.json()
                content = data["choices"][0]["message"].get("content", "").strip()
                clean = _strip_thinking(content)
                if clean:
                    return clean
        except Exception as e:
            pass
        return text

