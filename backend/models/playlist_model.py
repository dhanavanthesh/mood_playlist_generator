from typing import List, Dict, Tuple, Any
import numpy as np
from datetime import datetime, UTC
import logging

logger = logging.getLogger(__name__)

class PlaylistGenerator:
    def __init__(self):
        """Initialize the playlist generator"""
        self.modification_commands = {
            "energy": ["energetic", "upbeat", "intense"],
            "calm": ["calm", "peaceful", "relaxing"],
            "tempo": ["faster", "slower", "speed"],
            "mood": ["happier", "sadder", "darker", "lighter"],
            "acoustic": ["acoustic", "unplugged"],
            "dance": ["danceable", "groove", "rhythm"]
        }

    def generate(self, songs: List[Dict], features: Dict) -> List[Dict]:
        """Generate a playlist from songs based on mood features"""
        try:
            scored_songs = []
            for song in songs:
                if 'audio_features' not in song:
                    continue
                
                score = self._calculate_feature_match(song['audio_features'], features)
                scored_songs.append((song, score))
            
            scored_songs.sort(key=lambda x: x[1], reverse=True)
            selected_songs = [song for song, _ in scored_songs[:20]]  
            
            return selected_songs
        except Exception as e:
            logger.error(f"Error generating playlist: {e}")
            return []

    def parse_command(self, command: str) -> Tuple[str, Dict]:
        """Parse modification command into type and parameters"""
        command = command.lower()
        
        command_type = "unknown"
        parameters = {}
        
        for cmd_type, keywords in self.modification_commands.items():
            if any(keyword in command for keyword in keywords):
                command_type = cmd_type
                break
        
        if command_type == "energy":
            parameters["target"] = 0.8 if "more" in command else 0.2
        elif command_type == "tempo":
            parameters["direction"] = 1 if "faster" in command else -1
        elif command_type == "mood":
            parameters["valence"] = 0.8 if "happier" in command else 0.2
        elif command_type == "acoustic":
            parameters["acousticness"] = 0.8 if "more" in command else 0.2
        
        return command_type, parameters

    def apply_modification(self, tracks: List[Dict], command_type: str, parameters: Dict) -> List[Dict]:
        """Apply modification to playlist tracks"""
        try:
            if command_type == "unknown":
                return tracks
            
            modified_tracks = []
            for track in tracks:
                if 'audio_features' not in track:
                    continue
                
                score = self._calculate_modification_match(
                    track['audio_features'],
                    command_type,
                    parameters
                )
                modified_tracks.append((track, score))
            
            modified_tracks.sort(key=lambda x: x[1], reverse=True)
            return [track for track, _ in modified_tracks[:20]]
            
        except Exception as e:
            logger.error(f"Error modifying playlist: {e}")
            return tracks

    def _calculate_feature_match(self, audio_features: Dict, target_features: Dict) -> float:
        """Calculate how well a song matches target features"""
        score = 0.0
        for feature, target in target_features.items():
            if feature in audio_features:
                score += 1 - abs(audio_features[feature] - target)
        return score / len(target_features) if target_features else 0.0

    def _calculate_modification_match(self, audio_features: Dict, command_type: str, parameters: Dict) -> float:
        """Calculate how well a song matches modification criteria"""
        if command_type == "energy":
            return 1 - abs(audio_features.get('energy', 0) - parameters['target'])
        elif command_type == "tempo":
            normalized_tempo = audio_features.get('tempo', 120) / 200  
            return normalized_tempo if parameters['direction'] > 0 else 1 - normalized_tempo
        elif command_type == "mood":
            return 1 - abs(audio_features.get('valence', 0) - parameters['valence'])
        elif command_type == "acoustic":
            return 1 - abs(audio_features.get('acousticness', 0) - parameters['acousticness'])
        return 0.0