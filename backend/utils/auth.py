import os
import requests
from typing import Dict
from fastapi import HTTPException, status
import base64
import logging
from datetime import datetime, UTC

logger = logging.getLogger(__name__)



class SpotifyAuthManager:
    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
        self.auth_url = "https://accounts.spotify.com/api/token"

    def get_token(self, code: str) -> Dict:
        """Exchange authorization code for access token"""
        try:
            auth_header = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()

            response = requests.post(
                self.auth_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri
                },
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            
            response.raise_for_status()
            token_info = response.json()
            
            logger.info(f"Successfully exchanged code for token at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
            
            return token_info
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"HTTP error occurred: {http_err}"
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication error: {str(e)}"
            )
            
            
async def verify_token(token: str):
    """Verify Spotify access token"""
    try:
        response = requests.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token"
            )
            
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed"
        )