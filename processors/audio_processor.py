"""
Audio processor using Groq Whisper via direct API.
"""

from groq import Groq
from pathlib import Path
from typing import Optional

from models.schemas import RawInput, InputType
from config import Config
from processors.math_normalizer import MathNormalizer


class GroqWhisperProcessor:
    """
    Audio processor using Groq Whisper.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY required for audio transcription")

        self.client = Groq(api_key=self.api_key)
        self.model = Config.GROQ_WHISPER_MODEL

        # Spoken math → symbolic hints (PRE-normalization)
        self._math_phrases = {
            "square root of": "sqrt",
            "to the power of": "^",
            "raised to": "^",
            "squared": "^2",
            "cubed": "^3",
            "divided by": "/",
            "multiplied by": "*",
            "times": "*",
            "plus": "+",
            "minus": "-",
            "equals": "=",
            "equal to": "=",
            "limit as": "lim",
            "approaches": "->",
            "derivative of": "derivative",
            "integral of": "integral",
        }

    def process(self, audio_path: str) -> RawInput:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # File size guard (Groq limit)
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError("Audio file exceeds 25MB limit")

        try:
            with open(path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    file=(path.name, f.read()),
                    model=self.model,
                    language="en",
                    temperature=0.0,
                    response_format="verbose_json"
                )

            raw_text = transcription.text.strip()
            confidence = self._calculate_confidence(transcription)

            # 1️⃣ Spoken → symbolic hints
            converted = self._convert_math_phrases(raw_text)

            # 2️⃣ CENTRAL NORMALIZATION (CRITICAL)
            normalized = MathNormalizer.normalize(converted)

            return RawInput(
                type=InputType.AUDIO,
                content=normalized,
                original_content=raw_text,
                confidence=confidence,
                needs_review=confidence < Config.AUDIO_CONFIDENCE_THRESHOLD,
                metadata={
                    "raw_transcript": raw_text,
                    "normalized_text": normalized,
                    "model": self.model,
                    "processor": "groq_whisper"
                }
            )

        except Exception as e:
            return RawInput(
                type=InputType.AUDIO,
                content="",
                original_content=audio_path,
                confidence=0.0,
                needs_review=True,
                metadata={"error": str(e)}
            )

    def _calculate_confidence(self, transcription) -> float:
        segments = getattr(transcription, "segments", [])
        if not segments:
            return 0.85

        scores = []
        for seg in segments:
            avg_logprob = getattr(seg, "avg_logprob", -0.5)
            score = max(0.0, min(1.0, 1.0 + avg_logprob / 2))
            scores.append(score)

        return sum(scores) / len(scores)

    def _convert_math_phrases(self, text: str) -> str:
        text = text.lower()
        for phrase, symbol in sorted(
            self._math_phrases.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):
            text = text.replace(phrase, f" {symbol} ")
        return " ".join(text.split())


# Alias (used by app.py)
AudioProcessor = GroqWhisperProcessor


def get_audio_processor() -> GroqWhisperProcessor:
    if not Config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY required for audio processing")
    return GroqWhisperProcessor()
