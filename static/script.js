// Chat functionality
let currentModel = 'freelb';
let messageCount = 0;
let currentTheme = 'light';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadPredefinedPrompts();
    setupEventListeners();
    autoResizeTextarea();
    initializeTheme();
});

// Initialize theme from localStorage
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    currentTheme = savedTheme;
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon();
}

// Toggle theme
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', currentTheme);
    localStorage.setItem('theme', currentTheme);
    updateThemeIcon();
}

// Update theme icon
function updateThemeIcon() {
    const sunIcon = document.querySelector('.sun-icon');
    const moonIcon = document.querySelector('.moon-icon');
    
    if (currentTheme === 'dark') {
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
    } else {
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
    
    // Model switching
    document.querySelectorAll('.switch-btn').forEach(btn => {
        btn.addEventListener('click', (e) => switchModel(e.target.dataset.model));
    });

    // Chat form submission
    document.getElementById('chat-form').addEventListener('submit', (e) => {
        e.preventDefault();
        sendMessage();
    });

    // Textarea auto-resize and enter key handling
    const textarea = document.getElementById('user-input');
    textarea.addEventListener('input', autoResizeTextarea);
    textarea.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// Auto-resize textarea
function autoResizeTextarea() {
    const textarea = document.getElementById('user-input');
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

// Switch model
function switchModel(model) {
    currentModel = model;
    
    // Update UI
    document.querySelectorAll('.switch-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.model === model);
    });
    
    // Update stats
    const modelName = model === 'freelb' ? 'FreeLB' : 'Standard';
    document.getElementById('current-model').textContent = modelName;
    
    // Clear chat when switching models
    const messagesContainer = document.getElementById('messages');
    messagesContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🤖</div>
            <h2>Switched to ${modelName} Model</h2>
            <p>${model === 'freelb' ? 'This model is robust to adversarial attacks' : 'This model is vulnerable to adversarial attacks'}</p>
        </div>
    `;
    messageCount = 0;
    updateMessageCount();
}

// Load predefined prompts
async function loadPredefinedPrompts() {
    try {
        const response = await fetch('/predefined');
        const data = await response.json();
        
        // Populate normal prompts
        const normalContainer = document.getElementById('normal-prompts');
        console.log(normalContainer)
        normalContainer.innerHTML = data.normal.map(prompt => `
    <div class="prompt-item" data-text="${encodeURIComponent(prompt.text)}">
        ${escapeHtml(prompt.text)}
        <div class="prompt-category">${escapeHtml(prompt.category)}</div>
    </div>
`).join('');

const advContainer = document.getElementById('adversarial-prompts');
console.log(JSON.stringify(data.adversarial[3].text))
advContainer.innerHTML = data.adversarial.map(prompt => `
  <div class="prompt-item attack"
       data-text="${encodeURIComponent(prompt.text)}">
    ${escapeHtml(prompt.text)}
    <div class="prompt-category">${escapeHtml(prompt.category)}</div>
  </div>
`).join('');

document.querySelectorAll('.prompt-item').forEach(el => {
  el.addEventListener('click', () => {
    const text = decodeURIComponent(el.dataset.text);
    const isAttack = el.classList.contains('attack');
    usePrompt(text, isAttack);
  });
});



    } catch (error) {
        console.error('Error loading prompts:', error);
    }
}

// Use predefined prompt
function usePrompt(text, isAttack) {
    document.getElementById('user-input').value = text;
    sendMessage(isAttack);
}

// Send message
async function sendMessage(isAttack = false) {
    const input = document.getElementById('user-input');
    console.log(input)
    const message = input.value.trim();
    console.log(message)
    if (!message) return;
    
    // Disable input
    input.disabled = true;
    document.getElementById('send-btn').disabled = true;
    console.log('checking')
    // Clear input
    input.value = '';
    autoResizeTextarea();
    
    // Hide welcome message if present
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    
    // Add user message
    addMessage(message, 'user', isAttack);
    
    // Show typing indicator
    const typingId = showTypingIndicator();
    
    try {
        // Send to backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                prompt: message,
                model: currentModel
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        // Add bot response
        addMessage(data.response, 'assistant', false, data.is_gibberish);
        
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        addMessage('Sorry, there was an error processing your request.', 'assistant', false, false);
    }
    
    // Re-enable input
    input.disabled = false;
    document.getElementById('send-btn').disabled = false;
    input.focus();
    
    // Update count
    messageCount += 2;
    updateMessageCount();
}

// Add message to chat
function addMessage(text, role, isAttack = false, isGibberish = false) {
    const messagesContainer = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} ${isGibberish ? 'gibberish' : ''}`;
    
    let badge = '';
    if (isAttack) {
        badge = '<span class="message-badge badge-attack">⚔️ Attack</span>';
    } else if (role === 'assistant') {
        if (isGibberish) {
            badge = '<span class="message-badge badge-gibberish">❌ Gibberish</span>';
        } else {
            badge = '<span class="message-badge badge-authentic">✓ Authentic</span>';
        }
    }
    
    const header = role === 'user' 
        ? `<div class="message-header">You ${badge}</div>`
        : `<div class="message-header">Assistant ${badge}</div>`;
    
    messageDiv.innerHTML = `
        ${header}
        <div class="message-content">${escapeHtml(text)}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const messagesContainer = document.getElementById('messages');
    const typingDiv = document.createElement('div');
    const id = 'typing-' + Date.now();
    typingDiv.id = id;
    typingDiv.className = 'message assistant';
    typingDiv.innerHTML = `
        <div class="message-header">Assistant</div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return id;
}

// Remove typing indicator
function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// Update message count
function updateMessageCount() {
    document.getElementById('message-count').textContent = messageCount;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}