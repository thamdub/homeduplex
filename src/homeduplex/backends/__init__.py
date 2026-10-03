"""Backends: interfaces in `base`, protocol adapters in subpackages, and the factories that pick one from the
settings (the only place that names concrete adapters)."""

from homeduplex.backends.base import LanguageModel, SpeechToText, TextToSpeech
from homeduplex.config.schema import LLMSettings, OllamaChat, STTSettings, TTSSettings, WyomingSTT, WyomingTTS


def speech_to_text(settings: STTSettings) -> SpeechToText:
    if isinstance(settings, WyomingSTT):
        from homeduplex.backends.wyoming.stt import WyomingSpeechToText

        return WyomingSpeechToText(settings)
    from homeduplex.backends.openai.stt import OpenAISpeechToText

    return OpenAISpeechToText(settings)


def language_model(settings: LLMSettings) -> LanguageModel:
    if isinstance(settings, OllamaChat):
        from homeduplex.backends.ollama import OllamaChatModel

        return OllamaChatModel(settings)
    from homeduplex.backends.openai.chat import OpenAIChatModel

    return OpenAIChatModel(settings)


def text_to_speech(settings: TTSSettings) -> TextToSpeech:
    if isinstance(settings, WyomingTTS):
        from homeduplex.backends.wyoming.tts import WyomingTextToSpeech

        return WyomingTextToSpeech(settings)
    from homeduplex.backends.openai.tts import OpenAITextToSpeech

    return OpenAITextToSpeech(settings)
