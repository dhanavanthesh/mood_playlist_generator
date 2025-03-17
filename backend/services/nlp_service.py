from transformers import pipeline
from app.models.mood import MoodCategory
import os
from typing import Tuple

class MoodDetectionService:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model=os.getenv("MODEL_PATH"),
            tokenizer=os.getenv("MODEL_PATH")
        )
        
    def detect_mood(self, text: str) -> Tuple[MoodCategory, float]:
        """Detect mood from text input"""
        result = self.classifier(text)[0]
        label = self._map_label_to_mood(result["label"])
        confidence = result["score"]
        return label, confidence
    
    def _map_label_to_mood(self, label: str) -> MoodCategory:
        """Map model output to MoodCategory"""
        mapping = {
            "joy": MoodCategory.HAPPY,
            "sadness": MoodCategory.SAD,
            "energetic": MoodCategory.ENERGETIC,
            "calm": MoodCategory.CALM
        }
        return mapping.get(label, MoodCategory.HAPPY)