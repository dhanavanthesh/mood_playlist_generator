from app.models.mood import MoodPreferences, MoodFeatures
from app.services.spotify_service import SpotifyService
from typing import List, Dict
import numpy as np

class RecommendationService:
    def __init__(self, spotify_service: SpotifyService):
        self.spotify = spotify_service
        self.mood_features = {
            "happy": MoodFeatures(
                valence=0.7,
                energy=0.6,
                danceability=0.65,
                tempo=120,
                acousticness=0.3,
                instrumentalness=0.2
            ),
            "sad": MoodFeatures(
                valence=0.3,
                energy=0.4,
                danceability=0.4,
                tempo=90,
                acousticness=0.6,
                instrumentalness=0.3
            ),
            # Add other mood mappings
        }
    
    async def generate_playlist(
        self,
        mood_prefs: MoodPreferences,
        limit: int = 20
    ) -> List[Dict]:
        """Generate playlist based on mood preferences"""
        base_features = self.mood_features[mood_prefs.category]
        
        seed_tracks = await self.spotify.get_recommendations(
            limit=5,
            target_valence=base_features.valence,
            target_energy=base_features.energy,
            target_danceability=base_features.danceability
        )
        
        recommendations = await self.spotify.get_recommendations(
            seed_tracks=[track["id"] for track in seed_tracks],
            limit=limit,
            target_valence=base_features.valence,
            target_energy=base_features.energy,
            target_danceability=base_features.danceability,
            min_tempo=mood_prefs.min_tempo,
            max_tempo=mood_prefs.max_tempo
        )
        
        return recommendations