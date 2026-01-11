import re

class MathNormalizer:
    """
    Central math normalization for OCR, text, and audio inputs.
    """

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return text

        # Normalize unicode
        replacements = {
            "×": "*",
            "÷": "/",
            "−": "-",
            "√": "sqrt",
            "π": "pi",
            "²": "^2",
            "³": "^3",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        # ix2 → i*x^2
        text = re.sub(r"\b([a-zA-Z])x(\d+)\b", r"\1*x^\2", text)

        # x2 → x^2
        text = re.sub(r"\b([a-zA-Z])(\d+)\b", r"\1^\2", text)

        # 3x → 3*x
        text = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", text)

        # )( → )*(
        text = re.sub(r"\)\s*\(", r")*(", text)

        # sin x → sin(x)
        text = re.sub(r"\b(sin|cos|tan|log)\s+([a-zA-Z0-9]+)", r"\1(\2)", text)

        # Clean spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text
