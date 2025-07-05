// Chat functionality for the dental assistant
class ChatManager {
    constructor() {
        this.messagesContainer = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendMessage');
        this.clearButton = document.getElementById('clearChat');
        this.messages = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadChatHistory();
    }

    setupEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => {
            this.sendMessage();
        });

        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Clear chat
        this.clearButton.addEventListener('click', () => {
            this.clearChat();
        });

        // Auto-resize input
        this.messageInput.addEventListener('input', () => {
            this.autoResize();
        });
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Add user message to UI
        const userMessage = {
            role: 'user',
            message: message,
            timestamp: new Date().toISOString()
        };

        this.addMessage(userMessage);
        this.messageInput.value = '';
        this.autoResize();

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Send message to backend
            const response = await window.dentalApp.makeRequest('/chat', {
                method: 'POST',
                body: JSON.stringify({
                    message: message,
                    session_id: window.dentalApp.sessionId
                })
            });

            // Remove typing indicator
            this.hideTypingIndicator();

            if (response.status === 'success') {
                // Add assistant response to UI
                const assistantMessage = {
                    role: 'assistant',
                    message: response.response,
                    timestamp: new Date().toISOString()
                };

                this.addMessage(assistantMessage);
            } else {
                throw new Error('Failed to get response from assistant');
            }

        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            
            // Add error message
            const errorMessage = {
                role: 'assistant',
                message: 'Sorry, I encountered an error. Please try again.',
                timestamp: new Date().toISOString(),
                isError: true
            };

            this.addMessage(errorMessage);
            window.dentalApp.showNotification('Failed to send message. Please try again.', 'error');
        }
    }

    addMessage(messageObj) {
        this.messages.push(messageObj);
        this.renderMessage(messageObj);
        this.scrollToBottom();
    }

    renderMessage(messageObj) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${messageObj.role}`;
        
        if (messageObj.isError) {
            messageDiv.classList.add('error');
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.formatMessage(messageObj.message);

        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = window.dentalApp.formatTimestamp(messageObj.timestamp);

        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTime);

        this.messagesContainer.appendChild(messageDiv);
    }

    formatMessage(message) {
        // Convert URLs to clickable links
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        message = message.replace(urlRegex, '<a href="$1" target="_blank">$1</a>');
        
        // Convert line breaks to <br> tags
        message = message.replace(/\n/g, '<br>');
        
        // Format medical terms (make them bold)
        const medicalTerms = [
            'caries', 'periodontitis', 'gingivitis', 'pulpitis', 'endodontic',
            'orthodontic', 'prosthetic', 'periodontal', 'oral surgery', 'implant',
            'root canal', 'extraction', 'restoration', 'crown', 'bridge'
        ];
        
        medicalTerms.forEach(term => {
            const regex = new RegExp(`\\b${term}\\b`, 'gi');
            message = message.replace(regex, `<strong>    formatMessage(message) {
        // Convert URLs to clickable links
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        message = message.replace(urlRegex, '<a href="$1" target="_</strong>`);
        });

        return window.dentalApp.sanitizeHTML(message);
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing';
        typingDiv.id = 'typingIndicator';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'typing-dots';
        typingContent.innerHTML = '<span></span><span></span><span></span>';
        
        typingDiv.appendChild(typingContent);
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    autoResize() {
        const input = this.messageInput;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 100) + 'px';
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            this.messages = [];
            this.messagesContainer.innerHTML = '';
            window.dentalApp.sessionId = window.dentalApp.generateSessionId();
            window.dentalApp.displayWelcomeMessage();
            window.dentalApp.showNotification('Chat cleared successfully', 'success');
        }
    }

    async loadChatHistory() {
        try {
            const response = await window.dentalApp.makeRequest(`/chat-history/${window.dentalApp.sessionId}`);
            if (response.messages && response.messages.length > 0) {
                this.messages = response.messages;
                this.renderAllMessages();
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    renderAllMessages() {
        this.messagesContainer.innerHTML = '';
        this.messages.forEach(message => {
            this.renderMessage(message);
        });
        this.scrollToBottom();
    }

    exportChat() {
        const chatData = {
            sessionId: window.dentalApp.sessionId,
            messages: this.messages,
            exportDate: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dental-chat-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize chat manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
});

// Add typing indicator styles
const typingStyle = document.createElement('style');
typingStyle.textContent = `
    .typing-dots {
        display: flex;
        align-items: center;
        gap: 4px;
        padding: 10px;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #6b7280;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    .typing-dots span:nth-child(3) { animation-delay: 0s; }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .message.error {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
    }
`;
document.head.appendChild(typingStyle);
