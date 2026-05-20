document.addEventListener('DOMContentLoaded', () => {
    // Console Logger Utility
    const consoleEl = document.getElementById('console-output');
    const logConsole = (msg) => {
        if (consoleEl) {
            const time = new Date().toLocaleTimeString();
            const newMsg = document.createElement('div');
            newMsg.textContent = `[${time}] ${msg}`;
            consoleEl.appendChild(newMsg);
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }
    };

    // Initial Welcome
    logConsole("Robot Control Dashboard loaded.");

    // Layout Slider Logic
    const layoutSlider = document.getElementById('layout-slider');
    const layoutVal = document.getElementById('layout-val');
    
    const updateLayout = (widthPercent) => {
        document.documentElement.style.setProperty('--col-left', `${widthPercent}%`);
    };
    
    if (layoutSlider && layoutVal) {
        const savedWidth = localStorage.getItem('sidebarWidth');
        if (savedWidth) {
            layoutSlider.value = savedWidth;
            layoutVal.textContent = `${savedWidth}%`;
            updateLayout(savedWidth);
        }

        layoutSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            layoutVal.textContent = `${val}%`;
            updateLayout(val);
            localStorage.setItem('sidebarWidth', val);
        });
    }

    // Emergency Stop Handler
    const btnEstop = document.getElementById('btn-estop');
    if (btnEstop) {
        btnEstop.addEventListener('click', async () => {
            logConsole("EMERGENCY STOP TRIGGERED!");
            try {
                const res = await fetch('/api/stop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                logConsole(`System: ${data.message}`);
                
                // Visual feedback
                document.body.style.boxShadow = "inset 0 0 100px rgba(239, 68, 68, 0.5)";
                setTimeout(() => {
                    document.body.style.boxShadow = "none";
                }, 1000);

            } catch (error) {
                console.error("Failed to trigger emergency stop", error);
                logConsole("CRITICAL ERROR: Failed to send stop command!");
            }
        });
    }

    // Three-Position Tabs Slider Handler
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    


    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const targetTab = btn.dataset.tab;
            

            
            tabPanels.forEach(panel => {
                if (panel.id === `panel-${targetTab}`) {
                    panel.style.display = 'block';
                    panel.classList.add('active');
                } else {
                    panel.style.display = 'none';
                    panel.classList.remove('active');
                }
            });
            
            logConsole(`Activated mode: ${btn.textContent}`);
        });
    });

    // Local / Remote AI Switch Handler
    const aiToggle = document.getElementById('ai-toggle');
    const labelLocal = document.getElementById('label-local');
    const labelRemote = document.getElementById('label-remote');
    
    if (aiToggle) {
        aiToggle.addEventListener('change', async (e) => {
            const mode = e.target.checked ? 'remote' : 'local';
            if (mode === 'remote') {
                labelRemote.classList.add('active');
                labelLocal.classList.remove('active');
            } else {
                labelLocal.classList.add('active');
                labelRemote.classList.remove('active');
            }
            
            try {
                const res = await fetch('/api/set_ai_mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode })
                });
                await res.json();
                logConsole(`AI Mode changed to: ${mode.toUpperCase()} (${mode === 'local' ? 'Offline Local OpenCV' : 'VLM ChatGPT Service'})`);
            } catch (error) {
                console.error("Failed to set AI mode", error);
                logConsole("Error: Failed to change AI core mode");
            }
        });
    }

    // Local Planner Select Handler
    const plannerSelect = document.getElementById('planner-select');
    
    if (plannerSelect) {
        plannerSelect.addEventListener('change', async (e) => {
            const planner = e.target.value;
            try {
                const res = await fetch('/api/set_planner', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ planner })
                });
                await res.json();
                
                let plannerName = "Control Space Planner";
                if (planner === "teb") plannerName = "TEB Local Planner";
                if (planner === "custom_vector") plannerName = "Custom Vector Planner";
                
                logConsole(`Local Planner changed to: ${plannerName}`);
            } catch (error) {
                console.error("Failed to set planner", error);
                logConsole("Error: Failed to change active local planner");
            }
        });
    }

    // Detect Mode controls
    const btnDetect = document.getElementById('btn-detect');
    if (btnDetect) {
        btnDetect.addEventListener('click', async () => {
            logConsole("Triggering shopfront detection...");
            btnDetect.disabled = true;
            try {
                const res = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                logConsole(`System: ${data.message || 'Detection command sent.'}`);
            } catch (error) {
                console.error("Failed to trigger detection", error);
                logConsole("Error: Failed to connect to detection backend.");
            } finally {
                btnDetect.disabled = false;
            }
        });
    }

    const btnFindTag = document.getElementById('btn-find-tag');
    if (btnFindTag) {
        btnFindTag.addEventListener('click', async () => {
            logConsole("Toggling autonomous AprilTag search...");
            try {
                const res = await fetch('/api/find_tag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                logConsole(`System: ${data.message || 'Tag search toggled.'}`);
            } catch (error) {
                console.error("Failed to toggle tag search", error);
                logConsole("Error: Failed to connect to search backend.");
            }
        });
    }

    // Navigate to AprilTag handler
    const selectTag = document.getElementById('select-tag');
    const btnGoToTag = document.getElementById('btn-go-to-tag');
    
    if (btnGoToTag && selectTag) {
        btnGoToTag.addEventListener('click', async () => {
            const tagName = selectTag.value;
            if (!tagName) {
                alert("Please select a tag first!");
                return;
            }
            
            logConsole(`Commanding robot to navigate to tag: ${tagName}...`);
            btnGoToTag.disabled = true;
            
            try {
                const res = await fetch('/api/navigate_to_tag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tag_name: tagName })
                });
                const data = await res.json();
                if (res.ok) {
                    logConsole(`System: ${data.message}`);
                } else {
                    logConsole(`Error: ${data.message}`);
                }
            } catch (error) {
                console.error("Failed to navigate to tag", error);
                logConsole("Error: Failed to connect to navigation backend.");
            } finally {
                btnGoToTag.disabled = false;
            }
        });
    }

    // OpenAI API Key Submission Form Logic
    const apiKeyInput = document.getElementById('openai-api-key');
    const apiKeySubmit = document.getElementById('btn-submit-api-key');
    const apiKeyStatus = document.getElementById('api-key-status');

    if (apiKeySubmit && apiKeyInput) {
        apiKeySubmit.addEventListener('click', async () => {
            const apiKey = apiKeyInput.value.trim();
            if (!apiKey) {
                alert("Please enter a valid OpenAI API Key!");
                return;
            }
            try {
                apiKeyStatus.textContent = "Setting session key...";
                apiKeyStatus.className = "api-key-status updating";
                
                const response = await fetch('/api/set_api_key', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ api_key: apiKey })
                });
                const resData = await response.json();
                
                if (response.ok) {
                    apiKeyStatus.textContent = "Active session key loaded! (Remote AI enabled)";
                    apiKeyStatus.className = "api-key-status active-key";
                    apiKeyInput.value = ""; // Clear for safety
                    logConsole("System: OpenAI API Key dynamically updated for this session.");
                } else {
                    apiKeyStatus.textContent = `Error: ${resData.message}`;
                    apiKeyStatus.className = "api-key-status missing-key";
                }
            } catch (error) {
                console.error("Failed to submit API key", error);
                apiKeyStatus.textContent = "Error: Failed to connect to server.";
                apiKeyStatus.className = "api-key-status missing-key";
            }
        });
        
        // Submit on enter press
        apiKeyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                apiKeySubmit.click();
            }
        });
    }

    // Delivery Agent Chatbot Logic
    const chatInput = document.getElementById('chat-input');
    const chatSend = document.getElementById('btn-send-chat');
    const chatMessages = document.getElementById('chat-messages');
    
    const addChatMessage = (msg, sender) => {
        if (chatMessages) {
            const div = document.createElement('div');
            div.className = `msg ${sender}`;
            div.textContent = msg;
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    };
    
    const sendChatMessage = async () => {
        const text = chatInput.value.trim();
        if (!text) return;
        
        chatInput.value = '';
        addChatMessage(text, 'user');
        logConsole(`User prompt: "${text}"`);
        
        try {
            const res = await fetch('/api/delivery/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            addChatMessage(data.reply, 'bot');
            logConsole(`Delivery Agent: "${data.reply}"`);
        } catch (error) {
            console.error("Failed to send chat message", error);
            addChatMessage("Sorry, I am having trouble connecting to the delivery brain. Make sure the ROS nodes are running!", 'bot');
        }
    };
    
    if (chatSend) chatSend.addEventListener('click', sendChatMessage);
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }

    // Remote Control manual button publishers
    const buttons = {
        'btn-up': document.getElementById('btn-up'),
        'btn-down': document.getElementById('btn-down'),
        'btn-left': document.getElementById('btn-left'),
        'btn-right': document.getElementById('btn-right'),
        'btn-stop': document.getElementById('btn-stop')
    };

    let activeInterval = null;
    let currentCmd = { linear: 0, angular: 0 };

    const sendCommand = async (linear, angular) => {
        try {
            await fetch('/api/cmd_vel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ linear, angular })
            });
        } catch (error) {
            console.error("Failed to send manual command", error);
        }
    };

    const startCommand = (linear, angular) => {
        if (activeInterval) clearInterval(activeInterval);
        currentCmd = { linear, angular };
        sendCommand(linear, angular);
        activeInterval = setInterval(() => sendCommand(linear, angular), 100);
    };

    const stopCommand = () => {
        if (activeInterval) clearInterval(activeInterval);
        currentCmd = { linear: 0, angular: 0 };
        sendCommand(0, 0);
    };

    Object.values(buttons).forEach(btn => {
        if (!btn) return;

        const linear = parseFloat(btn.dataset.linear);
        const angular = parseFloat(btn.dataset.angular);

        btn.addEventListener('mousedown', () => {
            if (btn.id === 'btn-stop') {
                stopCommand();
            } else {
                startCommand(linear, angular);
            }
        });

        btn.addEventListener('mouseup', () => {
            if (btn.id !== 'btn-stop') {
                stopCommand();
            }
        });

        btn.addEventListener('mouseleave', () => {
            if (btn.id !== 'btn-stop' && currentCmd.linear === linear && currentCmd.angular === angular) {
                stopCommand();
            }
        });

        btn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (btn.id === 'btn-stop') {
                stopCommand();
            } else {
                startCommand(linear, angular);
            }
        }, {passive: false});

        btn.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (btn.id !== 'btn-stop') {
                stopCommand();
            }
        }, {passive: false});
    });

    // Keyboard support for WASD / Arrows
    document.addEventListener('keydown', (e) => {
        if (e.repeat) return;
        
        switch(e.key) {
            case 'ArrowUp':
            case 'w':
            case 'W':
                startCommand(0.5, 0);
                if (buttons['btn-up']) buttons['btn-up'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowDown':
            case 's':
            case 'S':
                startCommand(-0.5, 0);
                if (buttons['btn-down']) buttons['btn-down'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowLeft':
            case 'a':
            case 'A':
                startCommand(0, 1.0);
                if (buttons['btn-left']) buttons['btn-left'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                startCommand(0, -1.0);
                if (buttons['btn-right']) buttons['btn-right'].style.transform = 'translateY(2px)';
                break;
            case ' ':
                stopCommand();
                if (buttons['btn-stop']) buttons['btn-stop'].style.transform = 'translateY(2px)';
                break;
        }
    });

    document.addEventListener('keyup', (e) => {
        switch(e.key) {
            case 'ArrowUp':
            case 'w':
            case 'W':
                stopCommand();
                if (buttons['btn-up']) buttons['btn-up'].style.transform = '';
                break;
            case 'ArrowDown':
            case 's':
            case 'S':
                stopCommand();
                if (buttons['btn-down']) buttons['btn-down'].style.transform = '';
                break;
            case 'ArrowLeft':
            case 'a':
            case 'A':
                stopCommand();
                if (buttons['btn-left']) buttons['btn-left'].style.transform = '';
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                stopCommand();
                if (buttons['btn-right']) buttons['btn-right'].style.transform = '';
                break;
            case ' ':
                if (buttons['btn-stop']) buttons['btn-stop'].style.transform = '';
                break;
        }
    });

    // Telemetry and Map Legend Polling
    const robotPoseEl = document.getElementById('robot-pose');
    const shopsCountEl = document.getElementById('shops-count');
    const mapCoverageEl = document.getElementById('map-coverage');
    const tagsCountEl = document.getElementById('tags-count');
    const legendGrid = document.getElementById('legend-grid');

    const updateTelemetry = async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (robotPoseEl && data.x !== null) {
                robotPoseEl.textContent = `[X: ${data.x}, Y: ${data.y}, θ: ${data.yaw}]`;
            }
            if (shopsCountEl) {
                shopsCountEl.textContent = `${data.shops_detected} / 8`;
            }
            if (mapCoverageEl) {
                mapCoverageEl.textContent = `${data.explored_area} m²`;
            }
            if (tagsCountEl) {
                tagsCountEl.textContent = `${data.tags_detected} tags`;
            }
            
            // Dynamic Find Tag Button UI updates
            const btnFindTag = document.getElementById('btn-find-tag');
            if (btnFindTag) {
                if (data.searching_tag) {
                    if (!btnFindTag.classList.contains('searching')) {
                        btnFindTag.classList.add('searching');
                        btnFindTag.innerHTML = `
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 0.5rem; vertical-align: middle;"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
                            Stop Searching
                        `;
                    }
                } else {
                    if (btnFindTag.classList.contains('searching')) {
                        btnFindTag.classList.remove('searching');
                        btnFindTag.innerHTML = `
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 0.5rem; vertical-align: middle;"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                            Find Next Tag
                        `;
                    }
                }
            }
            
            // Dynamic Navigate to Tag Button UI updates
            const btnGoToTag = document.getElementById('btn-go-to-tag');
            if (btnGoToTag) {
                if (data.navigating_to_tag) {
                    btnGoToTag.textContent = "Navigating...";
                    btnGoToTag.style.background = "#d97706";
                } else {
                    btnGoToTag.textContent = "Go to Tag";
                    btnGoToTag.style.background = "#10b981";
                }
            }
            
            // Dynamic API Key Status Update
            if (apiKeyStatus && document.activeElement !== apiKeyInput) {
                if (data.has_api_key) {
                    apiKeyStatus.textContent = "Active session key loaded! (Remote AI enabled)";
                    apiKeyStatus.className = "api-key-status active-key";
                } else {
                    apiKeyStatus.textContent = "No session key loaded. (Remote AI disabled)";
                    apiKeyStatus.className = "api-key-status missing-key";
                }
            }

            // Dynamic Planner Select UI updates
            const plannerSelect = document.getElementById('planner-select');
            if (plannerSelect && data.local_planner && document.activeElement !== plannerSelect) {
                plannerSelect.value = data.local_planner;
            }
        } catch (error) {
            console.error("Failed to fetch telemetry status", error);
        }
    };

    const updateLogs = async () => {
        try {
            const response = await fetch('/api/logs');
            const data = await response.json();
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(log => {
                    logConsole(log);
                });
            }
        } catch (error) {
            console.error("Failed to fetch logs", error);
        }
    };

    const updateLegend = async () => {
        try {
            const res = await fetch('/api/semantic_map');
            const mapData = await res.json();
            
            for (let i = 0; i < 8; i++) {
                const item = legendGrid.querySelector(`.legend-item[data-idx="${i}"]`);
                if (!item) continue;
                
                if (mapData[i]) {
                    const store = mapData[i];
                    const sf = store.storefront;
                    const cat = store.category;
                    
                    let dotClass = 'unmapped';
                    if (cat === 'Café') dotClass = 'cafe';
                    else if (cat === 'Convenience store') dotClass = 'store';
                    else if (cat === 'Fast-food restaurant') dotClass = 'burger';
                    else if (cat === 'Pharmacy') dotClass = 'pharmacy';
                    
                    item.innerHTML = `<span class="legend-dot ${dotClass}"></span> S${i+1}: ${sf} (${cat})`;
                } else {
                    item.innerHTML = `<span class="legend-dot unmapped"></span> S${i+1}: Unmapped`;
                }
            }
        } catch (error) {
            console.error("Failed to fetch semantic map for legend", error);
        }
    };

    // Reset Robot & Map button click handler
    const btnReset = document.getElementById('btn-reset');
    if (btnReset) {
        btnReset.addEventListener('click', async () => {
            if (!confirm("Are you sure you want to reset the robot's pose, SLAM map database, and storefront classifications?")) {
                return;
            }
            
            btnReset.disabled = true;
            logConsole("Triggering global simulation, pose, and SLAM map reset...");
            
            try {
                const res = await fetch('/api/reset', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                
                logConsole("Reset successful: " + data.message);
                
                // Clear chatbot chat thread to initial greeting
                if (chatMessages) {
                    chatMessages.innerHTML = '<div class="msg bot">Hello! I am your AI delivery robot. Tell me what product or storefront you want me to find, and I will search the map or interpret signboards to bring it to you!</div>';
                }
                
                // Force instant telemetry refresh
                updateTelemetry();
                updateLegend();
            } catch (error) {
                console.error("Failed to reset robot and map", error);
                logConsole("Error: Reset command failed. Check server status.");
            } finally {
                btnReset.disabled = false;
            }
        });
    }

    // Trigger initial poll
    updateTelemetry();
    updateLegend();

    // Poll status every 500ms
    setInterval(updateTelemetry, 500);
    // Poll logs every 500ms
    setInterval(updateLogs, 500);
    // Poll semantic map legend every 1000ms
    setInterval(updateLegend, 1000);
});
