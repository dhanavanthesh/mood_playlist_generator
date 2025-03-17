import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, UTC
import os
from base64 import b64encode

logger = logging.getLogger(__name__)

class SpotifyService:
    def __init__(self):
        """Initialize Spotify service with API credentials"""
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.base_url = "https://api.spotify.com/v1"
        self.auth_url = "https://accounts.spotify.com/api/token"

    def get_user_profile(self, access_token: str) -> Dict:
        """Get user's Spotify profile"""
        try:
            response = requests.get(
                f"{self.base_url}/me",
                headers=self._get_auth_headers(access_token)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            raise

    def get_user_top_tracks(self, access_token: str) -> List[Dict]:
        """Get user's top tracks"""
        try:
            response = requests.get(
                f"{self.base_url}/me/top/tracks",
                headers=self._get_auth_headers(access_token),
                params={"limit": 50, "time_range": "medium_term"}
            )
            response.raise_for_status()
            return response.json()["items"]
        except Exception as e:
            logger.error(f"Error getting top tracks: {e}")
            return []

    def get_audio_features(self, access_token: str, track_id: str) -> Dict:
        """Get audio features for a track"""
        try:
            response = requests.get(
                f"{self.base_url}/audio-features/{track_id}",
                headers=self._get_auth_headers(access_token)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting audio features: {e}")
            return {}

    def fetch_mood_songs(self, access_token: str, features: Dict, context: str, seed_tracks: List[Dict]) -> List[Dict]:
        """Fetch songs matching mood features"""
        try:
            seed_track_ids = [track["id"] for track in seed_tracks[:5]]
            
            params = {
                "limit": 50,
                "seed_tracks": ",".join(seed_track_ids),
                "target_energy": features.get("energy", 0.5),
                "target_valence": features.get("valence", 0.5),
                "target_danceability": features.get("danceability", 0.5),
                "target_acousticness": features.get("acousticness", 0.5),
                "target_instrumentalness": features.get("instrumentalness", 0.5)
            }
            
            response = requests.get(
                f"{self.base_url}/recommendations",
                headers=self._get_auth_headers(access_token),
                params=params
            )
            response.raise_for_status()
            
            tracks = response.json()["tracks"]
            
            # Get audio features for all tracks
            for track in tracks:
                track["audio_features"] = self.get_audio_features(access_token, track["id"])
            
            return tracks
        except Exception as e:
            logger.error(f"Error fetching mood songs: {e}")
            return []

    def create_playlist(self, access_token: str, name: str, track_ids: List[str], description: str = "") -> str:
        """Create a new playlist and add tracks"""
        try:
            user_profile = self.get_user_profile(access_token)
            user_id = user_profile["id"]
            
            response = requests.post(
                f"{self.base_url}/users/{user_id}/playlists",
                headers=self._get_auth_headers(access_token),
                json={
                    "name": name,
                    "description": description,
                    "public": False
                }
            )
            response.raise_for_status()
            playlist_id = response.json()["id"]
            
            if track_ids:
                track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
                requests.post(
                    f"{self.base_url}/playlists/{playlist_id}/tracks",
                    headers=self._get_auth_headers(access_token),
                    json={"uris": track_uris}
                )
            
            return playlist_id
        except Exception as e:
            logger.error(f"Error creating playlist: {e}")
            raise

    def get_playlist_tracks(self, access_token: str, playlist_id: str) -> List[Dict]:
        """Get tracks from a playlist"""
        try:
            response = requests.get(
                f"{self.base_url}/playlists/{playlist_id}/tracks",
                headers=self._get_auth_headers(access_token)
            )
            response.raise_for_status()
            return [item["track"] for item in response.json()["items"]]
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {e}")
            return []

    def update_playlist(self, access_token: str, playlist_id: str, track_ids: List[str], description: str = "") -> bool:
        """Update playlist tracks"""
        try:
            # Update playlist tracks
            track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
            requests.put(
                f"{self.base_url}/playlists/{playlist_id}/tracks",
                headers=self._get_auth_headers(access_token),
                json={"uris": track_uris}
            )
            
            # Update description
            if description:
                requests.put(
                    f"{self.base_url}/playlists/{playlist_id}",
                    headers=self._get_auth_headers(access_token),
                    json={"description": description}
                )
            
            return True
        except Exception as e:
            logger.error(f"Error updating playlist: {e}")
            return False

    def _get_auth_headers(self, access_token: str) -> Dict:
        """Get headers for authenticated requests"""
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }