import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class MoodClassifier:
    def __init__(self):
        """Initialize the mood classifier with pre-trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moods = ["energetic", "calm", "happy", "melancholic", "focused"]
        self.contexts = ["workout", "relaxation", "party", "study", "focus"]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(self.moods)
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Using fallback classifier due to model loading error: {e}")
            self.tokenizer = None
            self.model = None

    def classify(self, text: str) -> Tuple[str, str]:
        """Classify the input text into mood and context"""
        # Fallback classification if model isn't loaded
        if self.model is None or self.tokenizer is None:
            return self._fallback_classify(text)
            
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                mood_idx = torch.argmax(predictions).item()
                
            context = self._determine_context(text)
            return self.moods[mood_idx], context
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return self._fallback_classify(text)

    def _fallback_classify(self, text: str) -> Tuple[str, str]:
        """Simple rule-based classification as fallback"""
        text = text.lower()
        
        mood_scores = {
            "energetic": sum(word in text for word in ["energy", "workout", "active", "pump"]),
            "calm": sum(word in text for word in ["relax", "peaceful", "calm", "chill"]),
            "happy": sum(word in text for word in ["happy", "joy", "fun", "party"]),
            "melancholic": sum(word in text for word in ["sad", "melancholic", "blue", "down"]),
            "focused": sum(word in text for word in ["focus", "concentrate", "study", "work"])
        }
        
        mood = max(mood_scores.items(), key=lambda x: x[1])[0]
        if all(score == 0 for score in mood_scores.values()):
            mood = "energetic"  # Default mood
            
        context = self._determine_context(text)
        return mood, context

    def _determine_context(self, text: str) -> str:
        """Determine the context from input text"""
        text = text.lower()
        
        context_scores = {
            "workout": sum(word in text for word in ["workout", "exercise", "gym", "running"]),
            "relaxation": sum(word in text for word in ["relax", "chill", "rest", "sleep"]),
            "party": sum(word in text for word in ["party", "dance", "club", "celebration"]),
            "study": sum(word in text for word in ["study", "learning", "homework", "reading"]),
            "focus": sum(word in text for word in ["focus", "work", "concentrate", "task"])
        }
        
        context = max(context_scores.items(), key=lambda x: x[1])[0]
        if all(score == 0 for score in context_scores.values()):
            context = "focus" 
            
        return context