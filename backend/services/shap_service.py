import numpy as np
from typing import Dict, List
import logging
import shap
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class ShapService:
    def __init__(self):
        """Initialize SHAP service"""
        self.feature_names = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo"
        ]
        
        self.background_data = self._initialize_background_data()

    def generate_explanation(self, song: Dict, features: Dict, mood: str) -> Dict:
        """Generate SHAP values explaining why a song matches the mood"""
        try:
            song_features = self._extract_features(song)
            target_features = self._convert_mood_to_features(mood)
            
            shap_values = self._calculate_shap_values(song_features, target_features)
            
            explanation = {}
            for feat_name, shap_value in zip(self.feature_names, shap_values):
                if abs(shap_value) > 0.01:
                    explanation[feat_name] = float(shap_value)
            
            return explanation
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {"error": "Could not generate explanation"}

    def _initialize_background_data(self) -> np.ndarray:
        """Initialize background data for SHAP explanations"""
        n_background = 100
        return np.random.rand(n_background, len(self.feature_names))

    def _extract_features(self, song: Dict) -> np.ndarray:
        """Extract audio features from song in correct order"""
        features = []
        audio_features = song.get('audio_features', {})
        
        for feature in self.feature_names:
            if feature == 'tempo':
                value = audio_features.get(feature, 120) / 200
            elif feature == 'loudness':
                # Normalize loudness from dB scale
                value = (audio_features.get(feature, -60) + 60) / 60
            else:
                value = audio_features.get(feature, 0.5)
            features.append(value)
        
        return np.array(features).reshape(1, -1)

    def _convert_mood_to_features(self, mood: str) -> np.ndarray:
        """Convert mood to target feature values"""
        mood_features = {
            "energetic": {
                "energy": 0.8,
                "tempo": 0.7,
                "valence": 0.7,
                "danceability": 0.7
            },
            "calm": {
                "energy": 0.3,
                "tempo": 0.4,
                "valence": 0.5,
                "acousticness": 0.7
            },
            "happy": {
                "energy": 0.7,
                "valence": 0.8,
                "danceability": 0.7
            },
            "melancholic": {
                "energy": 0.4,
                "valence": 0.3,
                "acousticness": 0.6
            },
            "focused": {
                "energy": 0.5,
                "instrumentalness": 0.6,
                "valence": 0.5
            }
        }
        
        target_features = mood_features.get(mood.lower(), {
            "energy": 0.5,
            "valence": 0.5,
            "danceability": 0.5
        })
        
        features = []
        for feature in self.feature_names:
            features.append(target_features.get(feature, 0.5))
        
        return np.array(features)

    def _calculate_shap_values(self, song_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for song features"""
        try:
            def model(x):
                return -np.sum(np.abs(x - target_features), axis=1)
            
            explainer = shap.KernelExplainer(model, self.background_data)
            
            shap_values = explainer.shap_values(song_features)
            
            return shap_values[0]
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return np.zeros(len(self.feature_names))