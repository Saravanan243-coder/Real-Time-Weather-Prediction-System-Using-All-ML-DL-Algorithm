document.addEventListener('DOMContentLoaded', () => {
    
    // 1. Rain Animation for Login Page
    const rainContainer = document.getElementById('rain-container');
    if (rainContainer) {
        createRain(rainContainer);
    }

    // 2. Prediction Form Handler
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }

    // 3. Auto-hide flash messages
    const flashMessages = document.querySelectorAll('.flash-message');
    if (flashMessages.length > 0) {
        setTimeout(() => {
            flashMessages.forEach(msg => {
                msg.style.opacity = '0';
                msg.style.transform = 'translateY(-20px)';
                msg.style.transition = 'all 0.5s ease';
                setTimeout(() => msg.remove(), 500);
            });
        }, 5000);
    }
});

// Function to generate CSS rain
function createRain(container) {
    const dropCount = 100;
    
    for (let i = 0; i < dropCount; i++) {
        const drop = document.createElement('div');
        drop.classList.add('raindrop');
        
        // Randomize properties
        const left = Math.floor(Math.random() * 100);
        const animDuration = 0.5 + Math.random();
        const animDelay = Math.random() * 5;
        const opacity = 0.1 + Math.random() * 0.4;
        
        drop.style.left = `${left}%`;
        drop.style.animationDuration = `${animDuration}s`;
        drop.style.animationDelay = `${animDelay}s`;
        drop.style.opacity = opacity;
        
        container.appendChild(drop);
    }
}

// Function to handle form submission and API call
async function handlePrediction(e) {
    e.preventDefault();
    
    const form = document.getElementById('prediction-form');
    const predictBtn = document.getElementById('predict-btn');
    const resultContainer = document.getElementById('prediction-result');
    const resultCard = document.getElementById('result-card');
    const resultIcon = document.getElementById('result-icon');
    const resultMessage = document.getElementById('result-message');
    const confidenceText = document.getElementById('confidence-text');
    const confidenceFill = document.getElementById('confidence-fill');
    
    // Original button text
    const originalBtnHTML = predictBtn.innerHTML;
    
    try {
        // Collect data
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            if (key !== 'model_name' && key !== 'RainToday') {
                // Try to parse numbers, keep strings for categorical
                const num = parseFloat(value);
                data[key] = isNaN(num) ? value : num;
            }
        }
        
        // Handle RainToday toggle specifically
        const rainToggle = document.getElementById('RainToday-toggle');
        if (rainToggle) {
            data['RainToday'] = rainToggle.checked ? "Yes" : "No";
        }
        
        const modelName = formData.get('model_name');
        
        // Loading state
        predictBtn.innerHTML = '<span class="spinner"></span> Predicting...';
        predictBtn.disabled = true;
        
        // Ensure result is hidden while loading
        resultContainer.classList.add('hidden');
        resultCard.className = 'result-card'; // reset classes
        confidenceFill.style.width = '0%';
        
        // API Call
        const response = await fetch(`/predict/${modelName}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Success - update UI
            resultContainer.classList.remove('hidden');
            
            // Scroll to result smoothly
            setTimeout(() => {
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
            
            const isRain = result.prediction === 'Yes';
            const confPercentage = Math.round(result.probability * 100);
            
            // Update content based on prediction
            if (isRain) {
                resultCard.classList.add('result-rain');
                resultIcon.innerHTML = '🌧️';
            } else {
                resultCard.classList.add('result-no-rain');
                resultIcon.innerHTML = '☀️';
            }
            
            resultMessage.innerText = result.message;
            confidenceText.innerText = `${confPercentage}%`;
            
            // Animate confidence bar
            setTimeout(() => {
                confidenceFill.style.width = `${confPercentage}%`;
                confidenceFill.style.background = isRain ? '#60a5fa' : '#fde047'; // light blue or yellow
            }, 300);
            
        } else {
            // Error handling
            alert(`Prediction Error: ${result.error}`);
        }
        
    } catch (error) {
        console.error("Prediction failed:", error);
        alert("Failed to connect to the prediction service.");
    } finally {
        // Reset button
        predictBtn.innerHTML = originalBtnHTML;
        predictBtn.disabled = false;
    }
}

// Function to handle Compass Direction selection
function selectDirection(btn) {
    // Get the parent container
    const container = btn.closest('.compass-select');
    
    // Remove active class from all buttons in this container
    const allBtns = container.querySelectorAll('.compass-btn');
    allBtns.forEach(b => b.classList.remove('active'));
    
    // Add active class to clicked button
    btn.classList.add('active');
    
    // Update the hidden input value
    const targetName = btn.getAttribute('data-target');
    const value = btn.getAttribute('data-value');
    
    const hiddenInput = container.querySelector(`input[name="${targetName}"]`);
    if (hiddenInput) {
        hiddenInput.value = value;
    }
    
    // Update the visual label
    const labelVal = document.getElementById(`${targetName}-val`);
    if (labelVal) {
        labelVal.textContent = value;
    }
}
