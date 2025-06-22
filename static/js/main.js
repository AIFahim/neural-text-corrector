// DOM elements
const inputText = document.getElementById('inputText');
const outputSection = document.getElementById('outputSection');
const outputText = document.getElementById('outputText');
const comparisonSection = document.getElementById('comparisonSection');
const originalText = document.getElementById('originalText');
const correctedText = document.getElementById('correctedText');
const correctBtn = document.getElementById('correctBtn');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const exampleBtn = document.getElementById('exampleBtn');
const wordCount = document.getElementById('wordCount');
const correctionCount = document.getElementById('correctionCount');
const processingTime = document.getElementById('processingTime');
const toast = document.getElementById('toast');

// Example sentences
let examples = [];
let currentExampleIndex = 0;

// Load examples on page load
fetch('/examples')
    .then(response => response.json())
    .then(data => {
        examples = data;
    });

// Event listeners
correctBtn.addEventListener('click', correctText);
clearBtn.addEventListener('click', clearText);
copyBtn.addEventListener('click', copyText);
exampleBtn.addEventListener('click', loadExample);
inputText.addEventListener('keydown', handleKeyDown);

// Handle Enter key with Ctrl/Cmd
function handleKeyDown(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        correctText();
    }
}

// Correct text function
async function correctText() {
    const text = inputText.value.trim();
    
    if (!text) {
        showToast('Please enter some text to correct');
        return;
    }
    
    // Show loading state
    correctBtn.disabled = true;
    correctBtn.querySelector('.btn-text').textContent = 'Processing...';
    correctBtn.querySelector('.loader').style.display = 'inline-block';
    
    try {
        const response = await fetch('/correct', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Display results
            displayResults(data);
        } else {
            showToast(data.error || 'An error occurred');
        }
    } catch (error) {
        showToast('Failed to connect to server');
        console.error('Error:', error);
    } finally {
        // Reset button state
        correctBtn.disabled = false;
        correctBtn.querySelector('.btn-text').textContent = 'Correct Text';
        correctBtn.querySelector('.loader').style.display = 'none';
    }
}

// Display results
function displayResults(data) {
    // Show output sections
    outputSection.style.display = 'block';
    comparisonSection.style.display = 'block';
    
    // Update output text
    outputText.textContent = data.corrected;
    
    // Update stats
    wordCount.textContent = data.word_count;
    correctionCount.textContent = data.corrections_made;
    processingTime.textContent = data.processing_time;
    
    // Show comparison with highlighted differences
    showComparison(data.original, data.corrected);
    
    // Scroll to results
    outputSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show comparison with differences highlighted
function showComparison(original, corrected) {
    const originalWords = original.split(' ');
    const correctedWords = corrected.split(' ');
    
    let originalHtml = '';
    let correctedHtml = '';
    
    const maxLength = Math.max(originalWords.length, correctedWords.length);
    
    for (let i = 0; i < maxLength; i++) {
        const origWord = originalWords[i] || '';
        const corrWord = correctedWords[i] || '';
        
        if (origWord !== corrWord) {
            originalHtml += `<span class="highlight">${origWord}</span> `;
            correctedHtml += `<span class="highlight">${corrWord}</span> `;
        } else {
            originalHtml += origWord + ' ';
            correctedHtml += corrWord + ' ';
        }
    }
    
    originalText.innerHTML = originalHtml.trim();
    correctedText.innerHTML = correctedHtml.trim();
}

// Clear text
function clearText() {
    inputText.value = '';
    outputSection.style.display = 'none';
    comparisonSection.style.display = 'none';
    inputText.focus();
}

// Copy corrected text
function copyText() {
    const text = outputText.textContent;
    
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text)
            .then(() => showToast('Copied to clipboard!'))
            .catch(() => fallbackCopy(text));
    } else {
        fallbackCopy(text);
    }
}

// Fallback copy method
function fallbackCopy(text) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
    showToast('Copied to clipboard!');
}

// Load example
function loadExample() {
    if (examples.length === 0) return;
    
    inputText.value = examples[currentExampleIndex];
    currentExampleIndex = (currentExampleIndex + 1) % examples.length;
    inputText.focus();
    
    showToast('Example loaded!');
}

// Show toast notification
function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Initialize
inputText.focus();