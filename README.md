# Mood-Based Playlist Generator using NLP

## Description
This project is a mood-based playlist generator that uses Natural Language Processing (NLP) to analyze user input, detect mood, and generate a personalized Spotify playlist. Users can request playlists using simple text inputs like:

💡 "I need focused music for studying"  
💡 "Play something energetic for my workout"  

The system extracts the mood, context, and audio preferences (e.g., tempo, energy, danceability) and finds suitable tracks on Spotify using their API.

## Features
✅ **Spotify OAuth Authentication** - Secure login via Spotify  
✅ **NLP Mood Analysis** - Understands user requests using text processing  
✅ **Personalized Playlists** - Generates playlists tailored to mood & context  
✅ **Real-time Track Selection** - Uses Spotify API for dynamic recommendations  
✅ **FastAPI Backend** - High-performance Python web API  

## Installation & Setup

### 1️⃣ Prerequisites
- Python 3.8+
- Spotify Developer Account ([Create One Here](https://developer.spotify.com/))
- Spotify API Credentials (Client ID & Secret)

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/dhanavanthesh/mood-playlist-generator.git
cd mood-playlist-generator
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add:

```ini
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8000/callback
JWT_SECRET=your_jwt_secret
```

### 5️⃣ Run the Application
```bash
uvicorn main:app --reload
```

### 6️⃣ Access the API
Open your browser and go to: [http://localhost:8000/docs](http://localhost:8000/docs)  
Test the endpoints using the interactive API documentation.

## Usage

### Example Requests
- "I need focused music for studying"
- "Play something energetic for my workout"

These inputs will be processed to generate playlists based on detected mood and context.

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License
This project is licensed under the MIT License.
