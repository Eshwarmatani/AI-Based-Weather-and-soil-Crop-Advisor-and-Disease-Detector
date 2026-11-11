/* ================================================================
   chatbot.js — AI Agriculture Assistant Frontend Controller
   AI Crop Advisor & Disease Detector
   ================================================================ */

(function () {
    const CHAT_HISTORY_STORAGE_KEY = 'agro_chat_history_v1';
    let chatHistory = [];
    let isBusy = false;
    let attachedFile = null;

    // DOM References
    let panel, triggerBtn, closeBtn, minimizeBtn, clearBtn;
    let confirmDialog, confirmCancel, confirmYes;
    let messagesContainer, typingIndicator, chatForm, chatInput, sendBtn;
    let attachBtn, imageInput, attachmentPreview, attachedImg, attachmentName, removeAttachmentBtn, modelSelect;
    let providerSubtitle, statusBadge;

    function init() {
        panel = document.getElementById('chatbotPanel');
        triggerBtn = document.getElementById('chatbotTriggerBtn');
        closeBtn = document.getElementById('chatbotCloseBtn');
        minimizeBtn = document.getElementById('chatbotMinimizeBtn');
        clearBtn = document.getElementById('chatbotClearBtn');
        confirmDialog = document.getElementById('chatbotConfirmDialog');
        confirmCancel = document.getElementById('chatbotConfirmCancel');
        confirmYes = document.getElementById('chatbotConfirmYes');
        messagesContainer = document.getElementById('chatbotMessages');
        typingIndicator = document.getElementById('chatbotTyping');
        chatForm = document.getElementById('chatbotForm');
        chatInput = document.getElementById('chatInput');
        sendBtn = document.getElementById('chatSendBtn');
        attachBtn = document.getElementById('chatAttachBtn');
        imageInput = document.getElementById('chatImageInput');
        attachmentPreview = document.getElementById('chatAttachmentPreview');
        attachedImg = document.getElementById('chatAttachedImg');
        attachmentName = document.getElementById('chatAttachmentName');
        removeAttachmentBtn = document.getElementById('chatRemoveAttachmentBtn');
        modelSelect = document.getElementById('chatModelSelect');
        providerSubtitle = document.getElementById('chatbotProviderSubtitle');
        statusBadge = document.getElementById('chatbotStatusBadge');

        if (!panel || !triggerBtn) return;

        // Load session history from sessionStorage
        try {
            const saved = sessionStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
            if (saved) {
                chatHistory = JSON.parse(saved);
                renderStoredMessages();
            }
        } catch (e) {
            chatHistory = [];
        }

        bindEvents();
        checkChatStatus();
    }

    function bindEvents() {
        // Toggle panel
        triggerBtn.addEventListener('click', () => {
            if (panel.classList.contains('open')) {
                if (panel.classList.contains('minimized')) {
                    panel.classList.remove('minimized');
                } else {
                    closePanel();
                }
            } else {
                openPanel();
            }
        });

        closeBtn.addEventListener('click', closePanel);

        minimizeBtn.addEventListener('click', () => {
            panel.classList.toggle('minimized');
        });

        // Clear chat
        clearBtn.addEventListener('click', () => {
            confirmDialog.style.display = 'block';
        });

        confirmCancel.addEventListener('click', () => {
            confirmDialog.style.display = 'none';
        });

        confirmYes.addEventListener('click', () => {
            confirmDialog.style.display = 'none';
            clearChatSession();
        });

        // Form submission
        chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            sendMessage();
        });

        // Textarea auto-resize & keyboard shortcuts
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });

        // Image Attachment
        attachBtn.addEventListener('click', () => {
            imageInput.click();
        });

        imageInput.addEventListener('change', (e) => {
            const file = e.target.files && e.target.files[0];
            if (file) {
                setAttachment(file);
            }
        });

        removeAttachmentBtn.addEventListener('click', clearAttachment);
    }

    function openPanel() {
        panel.classList.add('open');
        panel.classList.remove('minimized');
        panel.setAttribute('aria-hidden', 'false');
        setTimeout(() => chatInput.focus(), 200);
        scrollToBottom();
    }

    function closePanel() {
        panel.classList.remove('open');
        panel.setAttribute('aria-hidden', 'true');
    }

    function setAttachment(file) {
        attachedFile = file;
        attachmentName.textContent = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            attachedImg.src = e.target.result;
            attachmentPreview.style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }

    function clearAttachment() {
        attachedFile = null;
        imageInput.value = '';
        attachmentPreview.style.display = 'none';
        attachedImg.src = '';
    }

    function scrollToBottom() {
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    function formatTime() {
        const d = new Date();
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function formatMarkdown(text) {
        if (!text) return '';
        // Escape raw HTML first for safety
        let html = escapeHtml(text);

        // Bold: **text**
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Italic: *text*
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        // Inline code: `text`
        html = html.replace(/`([^`]+)`/g, '<code class="chat-code">$1</code>');

        // Links: [text](url)
        html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Line breaks & bullet points
        const lines = html.split('\n');
        let inList = false;
        let outLines = [];

        for (let line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('• ') || trimmed.startsWith('- ')) {
                if (!inList) {
                    outLines.push('<ul class="chat-bullet-list">');
                    inList = true;
                }
                outLines.push('<li>' + trimmed.substring(2) + '</li>');
            } else {
                if (inList) {
                    outLines.push('</ul>');
                    inList = false;
                }
                outLines.push(trimmed === '' ? '<div class="chat-space"></div>' : '<p>' + line + '</p>');
            }
        }
        if (inList) outLines.push('</ul>');

        return outLines.join('');
    }

    function appendMessage(role, content, imageUrl = null) {
        const isUser = role === 'user';
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-msg ${isUser ? 'chat-msg-user' : 'chat-msg-bot'}`;

        let imgHtml = '';
        if (imageUrl) {
            imgHtml = `<div class="chat-msg-img-wrap"><img src="${imageUrl}" alt="Uploaded leaf" class="chat-msg-thumb"></div>`;
        }

        if (isUser) {
            msgDiv.innerHTML = `
                <div class="chat-msg-bubble">
                    ${imgHtml}
                    ${content ? `<p>${escapeHtml(content)}</p>` : ''}
                    <span class="chat-msg-time">${formatTime()}</span>
                </div>
            `;
        } else {
            msgDiv.innerHTML = `
                <div class="chat-msg-avatar"><i class="fas fa-seedling"></i></div>
                <div class="chat-msg-bubble">
                    ${formatMarkdown(content)}
                    <span class="chat-msg-time">${formatTime()}</span>
                </div>
            `;
        }

        messagesContainer.appendChild(msgDiv);
        scrollToBottom();

        // Update stored history
        chatHistory.push({ role, content, time: new Date().toISOString() });
        try {
            sessionStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(chatHistory.slice(-20)));
        } catch (e) {}
    }

    function renderStoredMessages() {
        if (!chatHistory.length) return;
        // Keep initial welcome message, hide chips if there are messages
        const chips = document.getElementById('chatSuggestions');
        if (chips) chips.style.display = 'none';

        chatHistory.forEach(msg => {
            const isUser = msg.role === 'user';
            const msgDiv = document.createElement('div');
            msgDiv.className = `chat-msg ${isUser ? 'chat-msg-user' : 'chat-msg-bot'}`;
            if (isUser) {
                msgDiv.innerHTML = `
                    <div class="chat-msg-bubble">
                        <p>${escapeHtml(msg.content)}</p>
                    </div>
                `;
            } else {
                msgDiv.innerHTML = `
                    <div class="chat-msg-avatar"><i class="fas fa-seedling"></i></div>
                    <div class="chat-msg-bubble">
                        ${formatMarkdown(msg.content)}
                    </div>
                `;
            }
            messagesContainer.appendChild(msgDiv);
        });
    }

    function clearChatSession() {
        chatHistory = [];
        try {
            sessionStorage.removeItem(CHAT_HISTORY_STORAGE_KEY);
        } catch (e) {}

        // Reset message container to welcome message + chips
        messagesContainer.innerHTML = `
            <div class="chat-msg chat-msg-bot">
                <div class="chat-msg-avatar"><i class="fas fa-seedling"></i></div>
                <div class="chat-msg-bubble">
                    <p>Hello! I am your <strong>AI Agriculture Assistant</strong>. 🌾 How can I help you today?</p>
                    <p style="font-size:.82rem;margin-top:.4rem;color:var(--text-muted)">I can recommend crops based on your soil NPK &amp; climate, diagnose leaf diseases from uploaded photos, evaluate weather disease risks, or explain previous predictions in English, Urdu, or Roman Urdu.</p>
                </div>
            </div>
            <div class="chat-suggestions" id="chatSuggestions">
                <span class="chat-suggestions-title"><i class="fas fa-lightbulb"></i> Suggested Questions:</span>
                <div class="chat-chip-list">
                    <button type="button" class="chat-chip" onclick="window.sendSuggestedQuestion('Which crop is suitable for my soil?')">Which crop is suitable for my soil?</button>
                    <button type="button" class="chat-chip" onclick="window.sendSuggestedQuestion('How can I identify rice leaf diseases?')">How can I identify rice leaf diseases?</button>
                    <button type="button" class="chat-chip" onclick="window.sendSuggestedQuestion('What causes leaf blast?')">What causes leaf blast?</button>
                    <button type="button" class="chat-chip" onclick="window.sendSuggestedQuestion('How does humidity affect plant diseases?')">How does humidity affect plant diseases?</button>
                    <button type="button" class="chat-chip" onclick="window.sendSuggestedQuestion('How can I improve soil fertility?')">How can I improve soil fertility?</button>
                    <button type="button" class="chat-chip" onclick="window.sendSuggestedQuestion('What should I do if my plant has brown spots?')">What should I do if my plant has brown spots?</button>
                </div>
            </div>
        `;
        clearAttachment();
        chatInput.value = '';
        chatInput.style.height = 'auto';
    }

    async function sendMessage() {
        if (isBusy) return;
        const text = chatInput.value.trim();
        const fileToSend = attachedFile;

        if (!text && !fileToSend) return;

        // Hide suggestions once user starts chatting
        const chips = document.getElementById('chatSuggestions');
        if (chips) chips.style.display = 'none';

        // Display user message in UI
        let thumbUrl = null;
        if (fileToSend) {
            thumbUrl = URL.createObjectURL(fileToSend);
        }
        appendMessage('user', text, thumbUrl);

        // Clear input fields
        chatInput.value = '';
        chatInput.style.height = 'auto';
        clearAttachment();

        // Show typing indicator
        isBusy = true;
        typingIndicator.style.display = 'flex';
        sendBtn.disabled = true;
        scrollToBottom();

        try {
            let response;
            const targetModel = modelSelect ? modelSelect.value : 'garden';

            if (fileToSend) {
                const formData = new FormData();
                formData.append('image', fileToSend);
                formData.append('message', text || 'Diagnose the disease in this plant leaf photo.');
                formData.append('model_type', targetModel);
                formData.append('history', JSON.stringify(chatHistory.slice(-8)));

                response = await fetch('/api/chat', {
                    method: 'POST',
                    body: formData
                });
            } else {
                response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        history: chatHistory.slice(-8),
                        model_type: targetModel
                    })
                });
            }

            const data = await response.json();

            if (data.success && data.reply) {
                appendMessage('assistant', data.reply);
            } else {
                const errText = data.error || 'Unable to retrieve answer. Please try again.';
                appendMessage('assistant', `⚠️ **Error:** ${errText}`);
            }
        } catch (err) {
            appendMessage('assistant', `⚠️ **Connection Failure:** Unable to connect to the agricultural server (${err.message}). Ensure the Flask backend is running.`);
        } finally {
            isBusy = false;
            typingIndicator.style.display = 'none';
            sendBtn.disabled = false;
            scrollToBottom();
            chatInput.focus();
        }
    }

    async function checkChatStatus() {
        try {
            const resp = await fetch('/api/chat/status');
            const data = await resp.json();
            if (data.success && providerSubtitle) {
                providerSubtitle.textContent = data.active_provider;
                if (statusBadge) {
                    statusBadge.style.backgroundColor = 'var(--green-400)';
                }
            }
        } catch (e) {
            if (providerSubtitle) {
                providerSubtitle.textContent = 'Decision Support Engine';
            }
        }
    }

    // Suggested Question trigger
    window.sendSuggestedQuestion = function (questionText) {
        openPanel();
        chatInput.value = questionText;
        sendMessage();
    };

    document.addEventListener('DOMContentLoaded', init);
})();
