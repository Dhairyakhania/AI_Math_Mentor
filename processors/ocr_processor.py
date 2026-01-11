import easyocr
from PIL import Image
import numpy as np
from pathlib import Path
from models.schemas import RawInput, InputType
from config import Config
import re
from processors.math_normalizer import MathNormalizer


class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def process(self, image_path: str) -> RawInput:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)
        img_np = np.array(img)

        results = self.reader.readtext(img_np, detail=1)

        if not results:
            return RawInput(
                type=InputType.IMAGE,
                content="",
                original_content=image_path,
                confidence=0.0,
                needs_review=True,
                metadata={"error": "No text detected"}
            )

        texts, confidences = [], []

        for _, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        raw_text = " ".join(texts)
        avg_confidence = float(np.mean(confidences))

        # ✅ CENTRAL NORMALIZATION
        cleaned = MathNormalizer.normalize(raw_text)

        return RawInput(
            type=InputType.IMAGE,
            content=cleaned,
            original_content=image_path,
            confidence=avg_confidence,
            needs_review=avg_confidence < Config.OCR_CONFIDENCE_THRESHOLD,
            metadata={
                "raw_text": raw_text,
                "confidences": confidences
            }
        )
