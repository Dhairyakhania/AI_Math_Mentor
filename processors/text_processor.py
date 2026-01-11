from models.schemas import RawInput, InputType
import re
from processors.math_normalizer import MathNormalizer


class TextProcessor:
    def process(self, text: str) -> RawInput:
        if not text:
            text = ""

        text = " ".join(text.split())

        # Strip LaTeX
        text = self._process_latex(text)

        # ✅ CENTRAL NORMALIZATION
        cleaned = MathNormalizer.normalize(text)

        return RawInput(
            type=InputType.TEXT,
            content=cleaned,
            original_content=text,
            confidence=1.0,
            needs_review=False,
            metadata={}
        )

    def _process_latex(self, text: str) -> str:
        patterns = [
            (r"\$\$(.*?)\$\$", r"\1"),
            (r"\$(.*?)\$", r"\1"),
            (r"\\frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)"),
            (r"\\sqrt\{(.*?)\}", r"sqrt(\1)"),
            (r"\\cdot", "*"),
            (r"\\times", "*"),
            (r"\\div", "/"),
            (r"\\pi", "pi"),
            (r"\\theta", "theta"),
            (r"\^(\d)", r"^\1"),
            (r"\^\{(.*?)\}", r"^\1"),
        ]

        for pat, rep in patterns:
            text = re.sub(pat, rep, text)

        return text.strip()
