from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import requests
import base64
import json
import os
from datetime import datetime, timedelta
from jose import jwt
from typing import Optional, Dict, Any
import secrets
import urllib.parse
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import svgwrite
import time  
from datetime import datetime, timedelta

load_dotenv()

# CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
# os.makedirs(CACHE_DIR, exist_ok=True)

# CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
# os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variable for Hugging Face cache
# os.environ['HF_HOME'] = CACHE_DIR
# os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')

TEMPO_MIN = 60
TEMPO_MAX = 180
FEATURE_NAMES = ["energy", "tempo", "danceability", "valence"]

SAMPLE_SIZE = 100

EMOTION_MODEL = {
    "name": "j-hartmann/emotion-english-distilroberta-base",
    "revision": "main"
}

INTENT_MODEL = {
    "name": "facebook/bart-large-mnli",
    "revision": "c626438"
}


HUGGINGFACE_API_KEY = "api_key"  
MODEL_PATH = "j-hartmann/emotion-english-distilroberta-base"  
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_PATH}"


class MoodAnalysis(BaseModel):
    text: str

class PlaylistGeneration(BaseModel):
    mood: str
    context: Optional[str] = None
    features: Dict[str, float]

class AudioFeatures(BaseModel):
    energy: float
    tempo: float
    danceability: float
    valence: float


mood_classifier = None
intent_classifier = None
feature_explainer = None

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))

app = FastAPI(title="Spotify API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(FRONTEND_DIR, "templates"))


def create_mood_icons():
    """Create SVG mood icons if they don't exist"""
    icons_dir = os.path.join(FRONTEND_DIR, "static", "img", "mood-icons")
    os.makedirs(icons_dir, exist_ok=True)
    
    mood_colors = {
        "joy": "#FFD700",
        "sadness": "#4169E1",
        "anger": "#FF0000",
        "fear": "#800080",
        "love": "#FF69B4",
        "surprise": "#00FF00",
        "neutral": "#808080",
        "energetic": "#FFA500",
        "calm": "#87CEEB",
        "happy": "#FFD700",
        "sad": "#4169E1",
        "focused": "#98FB98"
    }
    
    for mood, color in mood_colors.items():
        svg_path = os.path.join(icons_dir, f"{mood}.svg")
        png_path = os.path.join(icons_dir, f"{mood}.png")
        
        if not os.path.exists(svg_path):
            dwg = svgwrite.Drawing(svg_path, size=('48px', '48px'))
            dwg.add(dwg.circle(center=(24, 24), r=20, fill=color))
            dwg.save()
            
            try:
                import cairosvg
                cairosvg.svg2png(url=svg_path, write_to=png_path)
            except ImportError:
                print("cairosvg not installed. Only SVG icons will be available.")



@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    print(f"\n=== New Request ===")
    print(f"Path: {request.url.path}")
    print(f"Method: {request.method}")
    
    if request.method == "POST":
        try:
            body = await request.body()
            print(f"Request body: {body.decode()}")
        except Exception as e:
            print(f"Could not read body: {e}")
    
    response = await call_next(request)
    print(f"Response status: {response.status_code}")
    return response

def generate_training_data(sample_size=100):
    """Generate more realistic training data for the model"""
    np.random.seed(42)
    
    energy = np.random.beta(2, 2, sample_size)  
    tempo = np.random.normal(120, 20, sample_size).clip(TEMPO_MIN, TEMPO_MAX)  
    danceability = np.random.beta(2, 2, sample_size)  
    valence = np.random.beta(2, 2, sample_size)  
    
    X = np.column_stack([energy, tempo, danceability, valence])
    
    y = (0.3 * energy + 
         0.2 * ((tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)) +
         0.3 * danceability +
         0.2 * valence)
    
    return X, y


def analyze_text_locally(text: str):
    """Analyze text locally when API fails"""
    text_lower = text.lower()
    
    # Keyword-based analysis
    keywords = {
        "energetic": ["energetic", "energy", "workout", "exercise", "pump", "hype", "upbeat"],
        "happy": ["happy", "joy", "excited", "fun", "cheerful", "positive"],
        "sad": ["sad", "down", "depression", "melancholy", "blue"],
        "calm": ["calm", "relaxed", "peaceful", "chill", "quiet", "mellow"],
        "focused": ["focus", "concentrate", "study", "work", "productive"]
    }
    
    # Check for direct mood mentions
    for mood, mood_keywords in keywords.items():
        if any(keyword in text_lower for keyword in mood_keywords):
            return {
                "label": mood,
                "score": 1.0
            }
    
    # Default to energetic if no mood is detected
    return {
        "label": "energetic",
        "score": 1.0
    }


class SpotifyClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    async def make_request(self, method: str, url: str, **kwargs):
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (401, 403):
                raise HTTPException(status_code=401, detail="Please login again")
            raise

    async def get_track_features(self, track_id: str):
        """Get audio features for a single track"""
        try:
            return await self.make_request(
                "GET",
                f"https://api.spotify.com/v1/audio-features/{track_id}"
            )
        except Exception:
            return None             

def analyze_mood_simple(text: str):
    """Simple keyword-based mood analysis"""
    text_lower = text.lower()
    
    # Keyword mappings
    mood_keywords = {
        "energetic": ["energetic", "energy", "workout", "exercise", "pump", "dance", "party", "upbeat", "fast"],
        "happy": ["happy", "joy", "fun", "cheerful", "excited", "positive", "great"],
        "sad": ["sad", "down", "depressed", "melancholy", "unhappy", "blue"],
        "calm": ["calm", "relaxed", "peaceful", "chill", "quiet", "mellow", "slow"],
        "focused": ["focus", "concentrate", "study", "work", "productive", "attention"]
    }
    
    for mood, keywords in mood_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return {
                "mood": mood,
                "confidence": 1.0,
                "features": MOOD_FEATURES[mood]
            }
    
    return {
        "mood": "energetic",
        "confidence": 1.0,
        "features": MOOD_FEATURES["energetic"]
    }
    
    
async def query_huggingface(text: str):
    """Query the Hugging Face API for emotion detection with fallback"""
    print("\n=== Analyzing Text ===")
    print(f"Input text: {text}")
    
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=10  
        )
        
        print(f"API Response Status: {response.status_code}")
        print(f"API Response: {response.text[:200]}")
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                emotions = result[0]
                max_emotion = max(emotions, key=lambda x: x['score'])
                return {
                    "label": max_emotion['label'],
                    "score": max_emotion['score']
                }
        
        print("API call failed, falling back to local analysis")
        return analyze_text_locally(text)
            
    except Exception as e:
        print(f"Error in Hugging Face API call: {str(e)}")
        print("Falling back to local analysis")
        return analyze_text_locally(text)      
          
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    try:
        os.makedirs(os.path.join(FRONTEND_DIR, "static", "img", "mood-icons"), exist_ok=True)
        
        create_mood_icons()
        
        print("Application initialized successfully!")
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise



def normalize_features(features):
    """Normalize audio features for SHAP analysis"""
    return np.array([
        features["energy"],
        (features["tempo"] - 60) / 120, 
        features["danceability"],
        features["valence"]
    ]).reshape(1, -1)





MOOD_FEATURES = {
    "energetic": {
        "energy": 0.8,
        "tempo": 140,
        "danceability": 0.7,
        "valence": 0.7
    },
    "calm": {
        "energy": 0.3,
        "tempo": 90,
        "danceability": 0.4,
        "valence": 0.5
    },
    "happy": {
        "energy": 0.7,
        "tempo": 120,
        "danceability": 0.8,
        "valence": 0.8
    },
    "sad": {
        "energy": 0.4,
        "tempo": 85,
        "danceability": 0.3,
        "valence": 0.3
    },
    "focused": {
        "energy": 0.5,
        "tempo": 110,
        "danceability": 0.4,
        "valence": 0.6
    },
    "neutral": {
        "energy": 0.5,
        "tempo": 110,
        "danceability": 0.5,
        "valence": 0.5
    }
}


CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
FRONTEND_URL = "http://localhost:8000"

state_store = {}

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None

class UserProfile(BaseModel):
    id: str
    display_name: Optional[str] = None
    external_urls: Dict[str, str]
    images: Optional[list] = None

async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        access_token = payload.get("access_token")
        if access_token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return access_token



@app.get("/api/test-mood")
async def test_mood():
    """Test the mood detection system"""
    test_cases = [
        "I need an energetic playlist for my workout",
        "I'm feeling sad today",
        "I need to focus on my work",
        "I want to party and dance",
        "I need to relax and chill"
    ]
    
    results = []
    for text in test_cases:
        try:
            result = await query_huggingface(text)
            results.append({
                "input": text,
                "result": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "input": text,
                "error": str(e),
                "success": False
            })
    
    return {
        "status": "success",
        "huggingface_api_status": "connected" if any(r["success"] for r in results) else "failed",
        "results": results
    }
    
   
   
   
@app.get("/api/test-connections")
async def test_connections(access_token: str = Depends(get_current_user)):
    """Test all required API connections"""
    results = {}
    headers = {"Authorization": f"Bearer {access_token}"}
    
    try:
        response = requests.get("https://api.spotify.com/v1/me", headers=headers)
        results["spotify_api"] = {
            "status": "success" if response.ok else "error",
            "status_code": response.status_code,
            "response": response.text[:200]
        }
    except Exception as e:
        results["spotify_api"] = {"status": "error", "error": str(e)}
    
    try:
        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": "test message"},
            timeout=30
        )
        results["huggingface_api"] = {
            "status": "success" if response.ok else "error",
            "status_code": response.status_code,
            "response": response.text[:200]
        }
    except Exception as e:
        results["huggingface_api"] = {"status": "error", "error": str(e)}
    
    return results

    
@app.get("/")
async def root(request: Request):
    access_token = request.cookies.get("access_token")
    user = None
    
    if access_token:
        try:
            payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
            spotify_token = payload.get("access_token")
            
            headers = {"Authorization": f"Bearer {spotify_token}"}
            response = requests.get("https://api.spotify.com/v1/me", headers=headers)
            
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text[:100]}...")
            
            if response.status_code == 200:
                user = response.json()
            else:
                print(f"Failed to get user info: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error verifying token: {str(e)}")
            response = templates.TemplateResponse("index.html", {
                "request": request,
                "user": None
            })
            response.delete_cookie("access_token")
            response.delete_cookie("refresh_token")
            return response
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": user
    })
@app.get("/login")
@app.get("/auth/spotify")
async def login():
    state = secrets.token_urlsafe(16)
    state_store[state] = datetime.now().timestamp()
    
    scope = " ".join([
        "user-read-private",
        "user-read-email",
        "playlist-read-private",
        "playlist-modify-public",
            "playlist-modify-private",  
        "user-library-read",
        "user-follow-read",
        "user-read-recently-played",
        "user-top-read"
    ])
    
    auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": CLIENT_ID,
        "scope": scope,
        "redirect_uri": REDIRECT_URI,
        "state": state,
        "show_dialog": True
    })
    
    return RedirectResponse(url=auth_url) 
    
    
    
    
    
    

@app.get("/callback")
async def callback(code: str = None, state: str = None, error: str = None, response: Response = None):
    if state not in state_store:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    
    state_store.pop(state, None)
    
    if error:
        raise HTTPException(status_code=400, detail=f"Authentication error: {error}")
    
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")
    
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI
    }
    
    try:
        token_response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
        token_response.raise_for_status()
        token_data = token_response.json()
        
        jwt_payload = {
            "access_token": token_data["access_token"],
            "exp": datetime.utcnow() + timedelta(seconds=token_data["expires_in"] - 60)
        }
        
        jwt_token = jwt.encode(
            jwt_payload,
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        
        response = RedirectResponse(url=FRONTEND_URL)
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            max_age=token_data["expires_in"] - 60,
            samesite="lax",
            path="/"  
        )
        
        if "refresh_token" in token_data:
            response.set_cookie(
                key="refresh_token",
                value=token_data["refresh_token"],
                httponly=True,
                max_age=365 * 24 * 60 * 60,  
                samesite="lax",
                path="/"  
            )
        
        return response
    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error exchanging code for token: {str(e)}")
    
@app.get("/api/me", response_model=UserProfile)
async def get_me(access_token: str = Depends(get_current_user)):
    """Get current user profile"""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get("https://api.spotify.com/v1/me", headers=headers)
        response.raise_for_status()
        print(f"Response: {response.json()}")
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error getting user profile: {str(e)}")

@app.get("/api/me/playlists")
async def get_my_playlists(
    access_token: str = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Get current user's playlists"""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get(
            f"https://api.spotify.com/v1/me/playlists?limit={limit}&offset={offset}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error getting playlists: {str(e)}")

@app.get("/api/playlists/{playlist_id}")
async def get_playlist(
    playlist_id: str,
    access_token: str = Depends(get_current_user)
):
    """Get details of a specific playlist"""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get(
            f"https://api.spotify.com/v1/playlists/{playlist_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error getting playlist: {str(e)}")

@app.get("/api/playlists/{playlist_id}/tracks")
async def get_playlist_tracks(
    playlist_id: str,
    access_token: str = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Get tracks from a specific playlist"""
    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        response = requests.get(
            f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit={limit}&offset={offset}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error getting playlist tracks: {str(e)}")

@app.get("/api/albums/{album_id}")
async def get_album(
    album_id: str,
    access_token: str = Depends(get_current_user),
    market: str = None
):
    """Get details of a specific album"""
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://api.spotify.com/v1/albums/{album_id}"
    if market:
        url += f"?market={market}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error getting album: {str(e)}")

@app.get("/logout")
async def logout():
    """Log out by clearing cookies and redirecting to the frontend"""
    response = RedirectResponse(url=FRONTEND_URL)
    
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    
    return response

@app.get("/library")
async def library_page(
    request: Request, 
    access_token: str = Depends(get_current_user)
):
    """Show user's library page with followed artists"""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        user_response = requests.get("https://api.spotify.com/v1/me", headers=headers)
        user_response.raise_for_status()
        user = user_response.json()
        
        followed_artists_response = requests.get(
            "https://api.spotify.com/v1/me/following?type=artist&limit=50",
            headers=headers
        )
        followed_artists_response.raise_for_status()
        followed_artists_data = followed_artists_response.json()
        
        library_data = {
            'followed_artists': followed_artists_data.get('artists', {}).get('items', [])
        }
        
        return templates.TemplateResponse(
            "library.html",
            {
                "request": request,
                "user": user,
                "library": library_data
            }
        )
        
    except Exception as e:
        print(f"Error in library_page: {str(e)}")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error_message": f"Error loading library: {str(e)}"
            },
            status_code=500
        )
  
         
@app.get("/profile")
async def profile_page(request: Request, access_token: str = Depends(get_current_user)):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get("https://api.spotify.com/v1/me", headers=headers)
    print(f"Response: {response.json()}")
    response.raise_for_status()
    user = response.json()
    return templates.TemplateResponse("profile.html", {"request": request, "user": user})

@app.get("/playlists")
async def playlists_page(
    request: Request, 
    access_token: str = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Show user playlists page and saved tracks"""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        user_response = requests.get("https://api.spotify.com/v1/me", headers=headers)
        user_response.raise_for_status()
        user = user_response.json()
        
        playlists_response = requests.get(
            f"https://api.spotify.com/v1/me/playlists?limit={limit}&offset={offset}",
            headers=headers
        )
        playlists_response.raise_for_status()
        playlists_data = playlists_response.json()
        
        saved_tracks_response = requests.get(
            "https://api.spotify.com/v1/me/tracks?limit=1",  
        )
        saved_tracks_response.raise_for_status()
        saved_tracks_data = saved_tracks_response.json()
        
        liked_songs_playlist = {
            'name': 'Liked Songs',
            'images': [{'url': 'https://misc.scdn.co/liked-songs/liked-songs-640.png'}],
            'tracks': {'total': saved_tracks_data.get('total', 0)},
            'type': 'liked_songs',
            'id': 'liked_songs'
        }
        
        playlists = {
            'items': [liked_songs_playlist] + playlists_data.get('items', []),
            'total': playlists_data.get('total', 0) + 1, 
            'limit': playlists_data.get('limit', limit),
            'offset': playlists_data.get('offset', offset),
            'next': playlists_data.get('next'),
            'previous': playlists_data.get('previous')
        }
        
        return templates.TemplateResponse("playlists.html", {
            "request": request,
            "user": user,
            "playlists": playlists
        })
        
    except requests.RequestException as e:
        print(f"Error in playlists_page: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting playlists: {str(e)}"
        )
        
        
        
@app.get("/refresh")
async def refresh_token(request: Request, response: Response):
    """Refresh the access token using the refresh token"""
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token available")
    
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    
    try:
        token_response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
        token_response.raise_for_status()
        token_data = token_response.json()
        
        jwt_payload = {
            "access_token": token_data["access_token"],
            "exp": datetime.utcnow() + timedelta(seconds=token_data["expires_in"] - 60)
        }
        jwt_token = jwt.encode(jwt_payload, SECRET_KEY, algorithm=ALGORITHM)
        
        response = RedirectResponse(url=FRONTEND_URL)
        response.set_cookie(
            key="access_token",
            value=jwt_token,
            httponly=True,
            max_age=token_data["expires_in"] - 60,
            samesite="lax",
            path="/" 
        )
        
        if "refresh_token" in token_data:
            response.set_cookie(
                key="refresh_token",
                value=token_data["refresh_token"],
                httponly=True,
                max_age=365 * 24 * 60 * 60, 
                samesite="lax",
                path="/"  
            )
        
        return response
    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing token: {str(e)}")



@app.get("/mood")
async def mood_page(request: Request, access_token: str = Depends(get_current_user)):
    """Render the mood-based playlist generator page"""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        user_response = requests.get("https://api.spotify.com/v1/me", headers=headers)
        user_response.raise_for_status()
        user = user_response.json()
        
        return templates.TemplateResponse("mood_generator.html", {
            "request": request,
            "user": user,
            "moods": list(MOOD_FEATURES.keys())
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/analyze-mood")
async def analyze_mood(analysis: MoodAnalysis):
    """Analyze text and return mood and features"""
    try:
        text = analysis.text.lower()
        
        mood_mapping = {
            "focused": ["focus", "study", "concentrate", "work"],
            "happy": ["happy", "joy", "excited", "fun"],
            "sad": ["sad", "down", "depressed", "unhappy"],
            "energetic": ["energy", "workout", "exercise", "pump"],
            "calm": ["calm", "relax", "peace", "quiet"]
        }
        
        detected_mood = "focused" 
        for mood, keywords in mood_mapping.items():
            if any(keyword in text for keyword in keywords):
                detected_mood = mood
                break
        
        features = MOOD_FEATURES.get(detected_mood, MOOD_FEATURES["focused"])
        
        context = None
        if any(word in text for word in ["study", "work", "homework", "learning"]):
            context = "study"
        elif any(word in text for word in ["gym", "workout", "exercise"]):
            context = "workout"
        
        return {
            "mood": detected_mood,
            "confidence": 1.0,
            "context": context,
            "features": features
        }
        
    except Exception as e:
        print(f"Error in analyze_mood: {str(e)}")
        return {
            "mood": "focused",
            "confidence": 1.0,
            "context": None,
            "features": MOOD_FEATURES["focused"]
        }     
        
@app.post("/api/generate-playlist")
async def generate_playlist(
    request: PlaylistGeneration,
    access_token: str = Depends(get_current_user)
):
    print("\n=== Playlist Generation Request ===")
    print(f"Mood: {request.mood}")
    
    try:
        spotify = SpotifyClient(access_token)
        
        user_data = await spotify.make_request(
            "GET",
            "https://api.spotify.com/v1/me"
        )
        user_id = user_data['id']
        
        print("\nStep 2: Searching for tracks")
        
        search_queries = {
            "focused": [
                "study instrumental piano",
                "focus instrumental",
                "concentration music",
                "study beats instrumental"
            ],
            "happy": [
                "happy instrumental",
                "upbeat instrumental",
                "positive piano",
                "joyful instrumental"
            ],
            "sad": [
                "sad piano instrumental",
                "melancholy instrumental",
                "emotional instrumental",
                "peaceful sad"
            ],
            "energetic": [
                "workout instrumental",
                "energy beats",
                "motivation instrumental",
                "power workout"
            ],
            "calm": [
                "relaxing piano",
                "calm instrumental",
                "peaceful ambient",
                "meditation music"
            ]
        }
        
        queries = search_queries.get(request.mood, [request.mood])
        all_tracks = []
        
        for query in queries:
            try:
                print(f"Searching for: {query}")
                response = await spotify.make_request(
                    "GET",
                    "https://api.spotify.com/v1/search",
                    params={
                        "q": query,
                        "type": "track",
                        "market": "US",
                        "limit": 10
                    }
                )
                
                if "tracks" in response and "items" in response["tracks"]:
                    tracks = [t for t in response["tracks"]["items"] if t.get("uri")]
                    all_tracks.extend(tracks)
                    print(f"Found {len(tracks)} tracks")
                    
            except Exception as e:
                print(f"Search error for '{query}': {e}")
                continue

        unique_tracks = list({t["uri"]: t for t in all_tracks}.values())
        selected_tracks = unique_tracks[:20]  
        
        print(f"\nSelected {len(selected_tracks)} tracks")

        if not selected_tracks:
            raise HTTPException(
                status_code=400,
                detail="No tracks found"
            )

        print("\nStep 3: Creating playlist")
        playlist_name = f"{request.mood.title()} Playlist"
        if request.context:
            playlist_name += f" for {request.context.title()}"
            
        playlist = await spotify.make_request(
            "POST",
            f"https://api.spotify.com/v1/users/{user_id}/playlists",
            json={
                "name": playlist_name,
                "description": f"Custom {request.mood} playlist generated {datetime.utcnow().strftime('%Y-%m-%d')}",
                "public": False
            }
        )

        print("\nStep 4: Adding tracks")
        track_uris = [track["uri"] for track in selected_tracks]
        
        await spotify.make_request(
            "POST",
            f"https://api.spotify.com/v1/playlists/{playlist['id']}/tracks",
            json={"uris": track_uris}
        )

        return {
            "status": "success",
            "playlist": playlist,
            "tracks": selected_tracks,
            "track_count": len(selected_tracks)
        }

    except Exception as e:
        print(f"\nError: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate playlist: {str(e)}"
        )
        
        
@app.post("/api/update-playlist")
async def update_playlist(
    playlist_id: str,
    features: AudioFeatures,
    access_token: str = Depends(get_current_user)
):
    """Update playlist based on adjusted features"""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        current_tracks_response = requests.get(
            f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
            headers=headers
        )
        current_tracks_response.raise_for_status()
        current_tracks = current_tracks_response.json()["items"]
        
        seed_tracks = [track["track"]["id"] for track in current_tracks[:5]]
        
        recommendation_params = {
            "limit": 20,
            "seed_tracks": ",".join(seed_tracks),
            "target_energy": features.energy,
            "target_tempo": features.tempo,
            "target_danceability": features.danceability,
            "target_valence": features.valence
        }
        
        return await generate_playlist(
            PlaylistGeneration(
                mood="custom",
                features=features.dict()
            ),
            access_token
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/remove-track/{playlist_id}/{track_id}")
async def remove_track(
    playlist_id: str,
    track_id: str,
    access_token: str = Depends(get_current_user)
):
    """Remove a track from the playlist"""
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        
        response = requests.delete(
            f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
            headers=headers,
            json={
                "tracks": [{"uri": f"spotify:track:{track_id}"}]
            }
        )
        response.raise_for_status()
        
        return await generate_playlist(
            PlaylistGeneration(
                mood="custom",
                features=MOOD_FEATURES["energetic"]  
            ),
            access_token
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
   