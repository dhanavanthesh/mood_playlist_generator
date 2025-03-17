// Handle playlist card hover animations
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth loading for images
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', () => {
            img.classList.add('opacity-100');
        });
        img.classList.add('opacity-0', 'transition-opacity', 'duration-300');
    });

    // Handle token refresh
    const checkTokenExpiry = async () => {
        try {
            const response = await fetch('/check-token');
            if (!response.ok) {
                await fetch('/refresh');
            }
        } catch (error) {
            console.error('Error checking token:', error);
        }
    };

    // Check token every 5 minutes
    setInterval(checkTokenExpiry, 5 * 60 * 1000);
});