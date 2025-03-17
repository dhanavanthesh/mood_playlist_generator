document.addEventListener('DOMContentLoaded', function() {
    const moodForm = document.getElementById('moodForm');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const moodResult = document.getElementById('moodResult');
    const errorContainer = document.getElementById('errorContainer');
    
    let currentPlaylist = null;

    const showError = (message) => {
        errorContainer.textContent = message;
        errorContainer.classList.remove('hidden');
        setTimeout(() => {
            errorContainer.classList.add('hidden');
        }, 5000);
    };

    const setLoading = (isLoading) => {
        loadingSection.classList.toggle('hidden', !isLoading);
        moodForm.classList.toggle('opacity-50', isLoading);
        moodForm.classList.toggle('pointer-events-none', isLoading);
    };

    moodForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const moodText = document.getElementById('moodText').value;

        try {
            setLoading(true);

            const moodResponse = await fetch('/api/analyze-mood', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: moodText })
            });

            if (!moodResponse.ok) {
                throw new Error('Failed to analyze mood');
            }

            const moodData = await moodResponse.json();
            displayMoodResult(moodData);

            const playlistResponse = await fetch('/api/generate-playlist', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mood: moodData.mood,
                    context: moodData.context,
                    features: moodData.features
                })
            });

            if (!playlistResponse.ok) {
                throw new Error('Failed to generate playlist');
            }

            const playlistData = await playlistResponse.json();
            currentPlaylist = playlistData;
            displayResults(playlistData);

        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'Error generating playlist');
        } finally {
            setLoading(false);
        }
    });

    function displayMoodResult(moodData) {
        const moodIcon = document.getElementById('moodIcon');
        const detectedMood = document.getElementById('detectedMood');
        const confidenceIndicator = document.getElementById('confidenceIndicator');
    
        moodIcon.src = `/static/img/mood-icons/${moodData.mood.toLowerCase()}.png`;
        moodIcon.onerror = function() {
            this.src = `/static/img/mood-icons/${moodData.mood.toLowerCase()}.svg`;
        };
    
        detectedMood.textContent = moodData.mood;
        const confidencePercentage = Math.round(moodData.confidence * 100);
        confidenceIndicator.textContent = `${confidencePercentage}% confidence`;
        confidenceIndicator.className = `text-sm ${
            confidencePercentage > 70 ? 'text-green-600' : 'text-yellow-600'
        }`;
    }

    function displayResults(data) {
        resultsSection.classList.remove('hidden');

        document.getElementById('playlistName').textContent = data.playlist.name;
        document.getElementById('playlistDescription').textContent = data.playlist.description;

        const tracksList = document.getElementById('tracksList');
        tracksList.innerHTML = data.tracks.map((track, index) => `
            <div class="py-4 flex items-center justify-between hover:bg-gray-50 rounded-lg transition-colors duration-200">
                <div class="flex items-center space-x-4">
                    <span class="text-gray-500 w-8">${index + 1}</span>
                    <img src="${track.album.images[2].url}" alt="" class="w-12 h-12 rounded-md">
                    <div>
                        <p class="font-medium">${track.name}</p>
                        <p class="text-gray-600 text-sm">${track.artists.map(a => a.name).join(', ')}</p>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <div class="text-sm text-gray-500">
                        ${formatShapExplanation(track.shap_values)}
                    </div>
                    <button 
                        onclick="removeTrack('${track.id}')"
                        class="text-red-500 hover:text-red-700 transition-colors duration-200"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                        </svg>
                    </button>
                </div>
            </div>
        `).join('');

        initializeFeatureControls(data.features);
        
        displayShapVisualization(data.tracks[0].shap_values);

        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    function formatShapExplanation(shapValues) {
        const top2Features = Object.entries(shapValues)
            .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
            .slice(0, 2);

        return top2Features.map(([feature, value]) => 
            `<span class="inline-flex items-center px-2 py-1 rounded-full text-xs ${
                value > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }">
                ${feature}: ${value > 0 ? '+' : ''}${value.toFixed(2)}
            </span>`
        ).join(' ');
    }

    function initializeFeatureControls(features) {
        const energySlider = document.getElementById('energySlider');
        const tempoSlider = document.getElementById('tempoSlider');
        const danceabilitySlider = document.getElementById('danceabilitySlider');
        const valenceSlider = document.getElementById('valenceSlider');

        if (!energySlider || !tempoSlider) return;

        energySlider.value = features.energy * 100;
        tempoSlider.value = features.tempo;
        if (danceabilitySlider) danceabilitySlider.value = features.danceability * 100;
        if (valenceSlider) valenceSlider.value = features.valence * 100;

        updateSliderValues();

        const debouncedUpdate = debounce(updatePlaylist, 500);
        
        [energySlider, tempoSlider, danceabilitySlider, valenceSlider].forEach(slider => {
            if (slider) {
                slider.addEventListener('input', updateSliderValues);
                slider.addEventListener('change', debouncedUpdate);
            }
        });
    }

    function updateSliderValues() {
        document.getElementById('energyValue').textContent = `${Math.round(energySlider.value)}%`;
        document.getElementById('tempoValue').textContent = `${Math.round(tempoSlider.value)} BPM`;
        if (danceabilitySlider) {
            document.getElementById('danceabilityValue').textContent = `${Math.round(danceabilitySlider.value)}%`;
        }
        if (valenceSlider) {
            document.getElementById('valenceValue').textContent = `${Math.round(valenceSlider.value)}%`;
        }
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    async function updatePlaylist() {
        if (!currentPlaylist) return;

        try {
            setLoading(true);
            const newFeatures = {
                energy: energySlider.value / 100,
                tempo: parseInt(tempoSlider.value),
                danceability: (danceabilitySlider?.value || 50) / 100,
                valence: (valenceSlider?.value || 50) / 100
            };

            const response = await fetch('/api/update-playlist', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    playlist_id: currentPlaylist.id,
                    features: newFeatures
                })
            });

            if (!response.ok) throw new Error('Failed to update playlist');
            
            const updatedData = await response.json();
            displayResults(updatedData);
        } catch (error) {
            showError(error.message);
        } finally {
            setLoading(false);
        }
    }

    function displayShapVisualization(shapValues) {
        const shapDiv = document.getElementById('shapExplanations');
        if (!shapDiv) return;

        const data = [{
            type: 'bar',
            x: Object.values(shapValues),
            y: Object.keys(shapValues).map(key => key.charAt(0).toUpperCase() + key.slice(1)),
            orientation: 'h',
            marker: {
                color: Object.values(shapValues).map(v => v > 0 ? '#1DB954' : '#FF4444')
            }
        }];

        const layout = {
            title: 'Feature Importance',
            xaxis: { 
                title: 'Impact on Recommendation',
                zeroline: true,
                zerolinecolor: '#969696',
                zerolinewidth: 1
            },
            height: 300,
            margin: { l: 150, r: 20, t: 40, b: 40 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                family: 'system-ui'
            }
        };

        Plotly.newPlot(shapDiv, data, layout, { displayModeBar: false });
    }

    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(element => {
        element.addEventListener('mouseenter', e => {
            const tooltip = document.createElement('div');
            tooltip.className = 'absolute bg-black text-white px-2 py-1 text-xs rounded transform -translate-y-full -translate-x-1/2 mt-1';
            tooltip.textContent = e.target.dataset.tooltip;
            document.body.appendChild(tooltip);
            
            const rect = e.target.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width / 2) + 'px';
            tooltip.style.top = rect.top - 5 + 'px';
            
            e.target.addEventListener('mouseleave', () => tooltip.remove(), { once: true });
        });
    });
});

async function removeTrack(trackId) {
    try {
        const response = await fetch(`/api/remove-track/${currentPlaylist.id}/${trackId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to remove track');
        
        const updatedData = await response.json();
        displayResults(updatedData);
    } catch (error) {
        showError(error.message);
    }
}