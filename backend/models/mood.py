from enum import Enum
from pydantic import BaseModel
from typing import List, Optional

class MoodCategory(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ENERGETIC = "energetic"
    CALM = "calm"
    FOCUSED = "focused"
    ROMANTIC = "romantic"

class MoodFeatures(BaseModel):
    valence: float
    energy: float
    danceability: float
    tempo: float
    acousticness: float
    instrumentalness: float

class MoodPreferences(BaseModel):
    category: MoodCategory
    context: Optional[str]
    min_tempo: Optional[float]
    max_tempo: Optional[float]
    min_energy: Optional[float]
    max_energy: Optional[float]
    min_valence: Optional[float]
    max_valence: Optional[float]