document.addEventListener('DOMContentLoaded', () => {
    // Setup web_video_server image streams depending on the current IP
    const host = window.location.hostname;
    // Camera streams disabled to avoid web UI bottleneck. Enable manually if needed.
    // document.getElementById('color-stream').src = `http://${host}:8080/stream?topic=/camera/color/image_annotated&type=mjpeg`;
    // document.getElementById('depth-stream').src = `http://${host}:8080/stream?topic=/camera/depth/image_color&type=mjpeg`;
    // If web_video_server is available, prefer streaming the local planner
    // via the /local_planner/image topic. The template still provides a
    // Flask fallback at /video_local_planner.
    try {
        const localEl = document.getElementById('local-planner-stream');
        if (localEl) {
            localEl.src = `http://${host}:8080/stream?topic=/local_planner/image&type=mjpeg`;
        }
    } catch (e) {
        // Ignore if element not present or assignment fails
        console.debug('Local planner stream not set via web_video_server', e);
    }

    const rosConsoleEl = document.getElementById('ros-console-output');
    const logRosConsole = (msg) => {
        if (rosConsoleEl) {
            const newMsg = document.createElement('div');
            newMsg.textContent = msg;
            // Style based on severity level prefix in message
            if (msg.includes('[WARN]')) {
                newMsg.style.color = '#fbbf24';
            } else if (msg.includes('[ERROR]') || msg.includes('[FATAL]')) {
                newMsg.style.color = '#ef4444';
            } else if (msg.includes('[DEBUG]')) {
                newMsg.style.color = '#94a3b8';
            } else {
                newMsg.style.color = '#38bdf8';
            }
            rosConsoleEl.appendChild(newMsg);
            
            // Limit lines to prevent DOM bloat
            while (rosConsoleEl.childNodes.length > 200) {
                rosConsoleEl.removeChild(rosConsoleEl.firstChild);
            }
            
            rosConsoleEl.scrollTop = rosConsoleEl.scrollHeight;
        }
    };
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

    // Viz Toggles (AprilTag / YOLO)
    const toggleApril = document.getElementById('toggle-apriltag');
    const toggleYolo = document.getElementById('toggle-yolo');

    const sendVizToggle = async (layer, enabled) => {
        try {
            await fetch('/api/viz_toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ layer, enabled })
            });
            logConsole(`Visualization: ${layer.toUpperCase()} marks ${enabled ? 'ON' : 'OFF'}`);
        } catch (e) {
            console.error(`Failed to toggle ${layer}`, e);
        }
    };

    if (toggleApril) {
        toggleApril.addEventListener('change', (e) => sendVizToggle('apriltag', e.target.checked));
    }
    if (toggleYolo) {
        toggleYolo.addEventListener('change', (e) => sendVizToggle('yolo', e.target.checked));
    }
    
    if (aiToggle) {
        aiToggle.addEventListener('change', async (e) => {
            const mode = e.target.checked ? 'remote' : 'local';
            if (mode === 'remote') {
                labelRemote.classList.add('active');
                labelLocal.classList.remove('active');
                const apiKeySection = document.getElementById('api-key-section');
                if (apiKeySection) apiKeySection.style.display = 'block';
            } else {
                labelLocal.classList.add('active');
                labelRemote.classList.remove('active');
                const apiKeySection = document.getElementById('api-key-section');
                if (apiKeySection) apiKeySection.style.display = 'none';
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

    // Go to mapped shop handler
    const selectMappedShop = document.getElementById('select-mapped-shop');
    const btnGoToShop = document.getElementById('btn-go-to-shop');
    if (btnGoToShop && selectMappedShop) {
        btnGoToShop.addEventListener('click', async () => {
            const shopName = selectMappedShop.value;
            if (!shopName) {
                alert("Please select a mapped shop first!");
                return;
            }
            
            const overshootVal = document.getElementById('overshoot-val');
            const overshoot_cm = overshootVal ? parseFloat(overshootVal.value) || 0 : 0;

            logConsole(`Commanding robot to navigate to mapped shop: ${shopName} (Overshoot: ${overshoot_cm}cm)...`);
            btnGoToShop.disabled = true;
            
            try {
                const res = await fetch('/api/goto_shop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: shopName, overshoot_cm })
                });
                const data = await res.json();
                if (res.ok && data.status === 'success') {
                    logConsole(`System: Navigation to ${shopName} started.`);
                } else {
                    logConsole(`Error: ${data.message}`);
                }
            } catch (error) {
                console.error("Failed to navigate to shop", error);
                logConsole("Error: Failed to connect to navigation backend.");
            } finally {
                btnGoToShop.disabled = false;
            }
        });
    }

    const btnDeleteShop = document.getElementById('btn-delete-shop');
    if (btnDeleteShop && selectMappedShop) {
        btnDeleteShop.addEventListener('click', async () => {
            const shopName = selectMappedShop.value;
            if (!shopName) {
                alert("Please select a mapped shop to delete!");
                return;
            }
            
            logConsole(`Deleting mapped shop: ${shopName}...`);
            btnDeleteShop.disabled = true;
            
            try {
                const res = await fetch('/api/delete_shop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: shopName })
                });
                const data = await res.json();
                if (res.ok) {
                    logConsole(`System: Deleted ${shopName}`);
                } else {
                    logConsole(`Error: ${data.message}`);
                }
            } catch (error) {
                console.error("Failed to delete shop", error);
                logConsole("Error: Failed to connect to backend.");
            } finally {
                btnDeleteShop.disabled = false;
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

    // Overshoot Value Apply Handler
    const overshootVal = document.getElementById('overshoot-val');
    const btnApplyOvershoot = document.getElementById('btn-apply-overshoot');
    if (btnApplyOvershoot && overshootVal) {
        btnApplyOvershoot.addEventListener('click', async () => {
            const overshootValNum = parseFloat(overshootVal.value) || 0;
            try {
                btnApplyOvershoot.disabled = true;
                const response = await fetch('/api/set_overshoot', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ overshoot_cm: overshootValNum })
                });
                const resData = await response.json();
                if (response.ok) {
                    logConsole(`System: Approach overshoot set to ${overshootValNum} cm.`);
                } else {
                    logConsole(`Error: ${resData.message}`);
                }
            } catch (error) {
                console.error("Failed to apply overshoot", error);
                logConsole("Error: Failed to connect to server.");
            } finally {
                btnApplyOvershoot.disabled = false;
            }
        });
    }

    // Clear Active Todo List Handler
    const btnClearTodo = document.getElementById('btn-clear-todo');
    if (btnClearTodo) {
        btnClearTodo.addEventListener('click', async () => {
            try {
                btnClearTodo.disabled = true;
                const response = await fetch('/api/delivery/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                if (response.ok) {
                    logConsole("System: Active delivery plan cleared.");
                    const todoSection = document.getElementById('todo-section');
                    if (todoSection) {
                        todoSection.style.display = 'none';
                    }
                }
            } catch (error) {
                console.error("Failed to clear delivery plan", error);
            } finally {
                btnClearTodo.disabled = false;
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
        const dummyHistory = [
            ...Array.from(chatMessages.children).map(child => ({
                sender: child.classList.contains('user') ? 'user' : 'bot',
                text: child.textContent
            })),
            { sender: 'user', text }
        ];
        lastChatCount = 0; // force re-render
        syncChat(dummyHistory);
        logConsole(`User prompt: "${text}"`);
        
        const overshootVal = document.getElementById('overshoot-val');
        const overshoot_cm = overshootVal ? parseFloat(overshootVal.value) || 0 : 0;
        
        try {
            const res = await fetch('/api/delivery/send', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, overshoot_cm })
            });
            const data = await res.json();
            // Force quick update
            updateTelemetry();
        } catch (error) {
            console.error("Failed to send chat message", error);
            const errorHistory = [
                ...dummyHistory,
                { sender: 'bot', text: "Sorry, I am having trouble connecting to the delivery brain. Make sure the ROS nodes are running!" }
            ];
            lastChatCount = 0; // force re-render
            syncChat(errorHistory);
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
    let currentMaxSpeed = 0.09;  // m/s
    let currentMaxAngular = 0.3;   // rad/s

    const buttons = {
        'btn-up':           document.getElementById('btn-up'),
        'btn-down':         document.getElementById('btn-down'),
        'btn-left':         document.getElementById('btn-left'),
        'btn-right':        document.getElementById('btn-right'),
        'btn-stop':         document.getElementById('btn-stop'),
        'btn-strafe-left':  document.getElementById('btn-strafe-left'),
        'btn-strafe-right': document.getElementById('btn-strafe-right')
    };

    let activeInterval = null;
    let currentCmd = { linear: 0, strafe: 0, angular: 0 };

    // strafe maps to linear.y in Twist (mecanum sideways)
    const sendCommand = async (linear, angular, strafe = 0) => {
        try {
            await fetch('/api/cmd_vel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ linear, angular, strafe })
            });
        } catch (error) {
            console.error("Failed to send manual command", error);
        }
    };

    const startCommand = (linear, angular, strafe = 0) => {
        if (activeInterval) clearInterval(activeInterval);
        currentCmd = { linear, angular, strafe };
        sendCommand(linear, angular, strafe);
        activeInterval = setInterval(() => sendCommand(linear, angular, strafe), 100);
    };

    const stopCommand = () => {
        if (activeInterval) clearInterval(activeInterval);
        currentCmd = { linear: 0, angular: 0, strafe: 0 };
        sendCommand(0, 0, 0);
    };

    Object.values(buttons).forEach(btn => {
        if (!btn) return;

        btn.addEventListener('mousedown', () => {
            if (btn.id === 'btn-stop') {
                stopCommand();
            } else {
                const linear = parseFloat(btn.dataset.linear  || 0);
                const angular = parseFloat(btn.dataset.angular || 0);
                const strafe  = parseFloat(btn.dataset.strafe  || 0);
                startCommand(linear, angular, strafe);
            }
        });

        btn.addEventListener('mouseup', () => {
            if (btn.id !== 'btn-stop') stopCommand();
        });

        btn.addEventListener('mouseleave', () => {
            if (btn.id !== 'btn-stop') stopCommand();
        });

        btn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (btn.id === 'btn-stop') {
                stopCommand();
            } else {
                const linear = parseFloat(btn.dataset.linear  || 0);
                const angular = parseFloat(btn.dataset.angular || 0);
                const strafe  = parseFloat(btn.dataset.strafe  || 0);
                startCommand(linear, angular, strafe);
            }
        }, {passive: false});

        btn.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (btn.id !== 'btn-stop') stopCommand();
        }, {passive: false});
    });

    // Speed Slider Dynamic Event Handling
    const speedSlider = document.getElementById('speed-limit-slider');
    const speedLimitVal = document.getElementById('speed-limit-val');

    if (speedSlider) {
        speedSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            speedLimitVal.textContent = `${val.toFixed(2)} m/s`;
            
            currentMaxSpeed = val;
            currentMaxAngular = val * 5.0; // Maintain same speed ratio (0.06 -> 0.3)
            
            // Dynamically update D-pad buttons dataset values
            if (buttons['btn-up']) {
                buttons['btn-up'].dataset.linear = currentMaxSpeed;
            }
            if (buttons['btn-down']) {
                buttons['btn-down'].dataset.linear = -currentMaxSpeed;
            }
            if (buttons['btn-left']) {
                buttons['btn-left'].dataset.angular = currentMaxAngular;
            }
            if (buttons['btn-right']) {
                buttons['btn-right'].dataset.angular = -currentMaxAngular;
            }
            if (buttons['btn-strafe-left']) {
                buttons['btn-strafe-left'].dataset.strafe = currentMaxSpeed;
            }
            if (buttons['btn-strafe-right']) {
                buttons['btn-strafe-right'].dataset.strafe = -currentMaxSpeed;
            }
        });

        speedSlider.addEventListener('change', async (e) => {
            const val = parseFloat(e.target.value);
            const valTheta = val * 5.0;
            try {
                await fetch('/api/set_max_vel', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ max_vel: val, max_vel_theta: valTheta })
                });
            } catch (error) {
                console.error("Failed to sync speed limit to backend", error);
            }
        });
    }

    // Keyboard support for WASD / Arrows / QE strafe
    document.addEventListener('keydown', (e) => {
        if (e.repeat) return;
        switch(e.key) {
            case 'ArrowUp':
            case 'w': case 'W':
                startCommand(currentMaxSpeed, 0, 0);
                if (buttons['btn-up']) buttons['btn-up'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowDown':
            case 's': case 'S':
                startCommand(-currentMaxSpeed, 0, 0);
                if (buttons['btn-down']) buttons['btn-down'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowLeft':
            case 'a': case 'A':
                startCommand(0, currentMaxAngular, 0);  // rotate left
                if (buttons['btn-left']) buttons['btn-left'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowRight':
            case 'd': case 'D':
                startCommand(0, -currentMaxAngular, 0); // rotate right
                if (buttons['btn-right']) buttons['btn-right'].style.transform = 'translateY(2px)';
                break;
            case 'q': case 'Q':
                startCommand(0, 0, currentMaxSpeed);   // strafe left
                if (buttons['btn-strafe-left']) buttons['btn-strafe-left'].style.transform = 'translateY(2px)';
                break;
            case 'e': case 'E':
                startCommand(0, 0, -currentMaxSpeed);  // strafe right
                if (buttons['btn-strafe-right']) buttons['btn-strafe-right'].style.transform = 'translateY(2px)';
                break;
            case ' ':
                stopCommand();
                if (buttons['btn-stop']) buttons['btn-stop'].style.transform = 'translateY(2px)';
                break;
        }
    });

    document.addEventListener('keyup', (e) => {
        switch(e.key) {
            case 'ArrowUp':    case 'w': case 'W': stopCommand(); if (buttons['btn-up'])           buttons['btn-up'].style.transform = ''; break;
            case 'ArrowDown':  case 's': case 'S': stopCommand(); if (buttons['btn-down'])         buttons['btn-down'].style.transform = ''; break;
            case 'ArrowLeft':  case 'a': case 'A': stopCommand(); if (buttons['btn-left'])         buttons['btn-left'].style.transform = ''; break;
            case 'ArrowRight': case 'd': case 'D': stopCommand(); if (buttons['btn-right'])        buttons['btn-right'].style.transform = ''; break;
            case 'q': case 'Q': stopCommand(); if (buttons['btn-strafe-left'])  buttons['btn-strafe-left'].style.transform = ''; break;
            case 'e': case 'E': stopCommand(); if (buttons['btn-strafe-right']) buttons['btn-strafe-right'].style.transform = ''; break;
            case ' ': if (buttons['btn-stop']) buttons['btn-stop'].style.transform = ''; break;
        }
    });

    // Telemetry and Map Legend Polling
    const robotPoseEl = document.getElementById('robot-pose');
    const shopsCountEl = document.getElementById('shops-count');
    const mapCoverageEl = document.getElementById('map-coverage');
    const tasksCountEl = document.getElementById('tasks-fulfilled');

    let lastChatCount = 0;
    const syncChat = (messages) => {
        if (!chatMessages || !messages) return;
        if (messages.length !== lastChatCount) {
            chatMessages.innerHTML = '';
            messages.forEach(msg => {
                const div = document.createElement('div');
                div.className = `msg ${msg.sender}`;
                div.textContent = msg.text;
                chatMessages.appendChild(div);
            });
            chatMessages.scrollTop = chatMessages.scrollHeight;
            lastChatCount = messages.length;
        }
    };

    const todoSection = document.getElementById('todo-section');
    const todoStatus = document.getElementById('todo-task-status');
    const todoContainer = document.getElementById('todo-list-container');

    const updateTodoList = (todoList) => {
        if (!todoSection || !todoStatus || !todoContainer) return;
        
        if (!todoList || todoList.status === 'idle') {
            todoSection.style.display = 'none';
            return;
        }
        
        todoSection.style.display = 'block';
        
        // Update general status
        if (todoList.status === 'completed') {
            todoStatus.textContent = 'Completed';
            todoStatus.style.background = 'rgba(16, 185, 129, 0.2)';
            todoStatus.style.color = '#10b981';
        } else {
            todoStatus.textContent = 'In Progress';
            todoStatus.style.background = 'rgba(59, 130, 246, 0.2)';
            todoStatus.style.color = '#60a5fa';
        }
        
        todoContainer.innerHTML = '';
        todoList.stores.forEach(store => {
            const storeDiv = document.createElement('div');
            
            // Determine styles based on status
            let borderStyle = '2px solid #334155';
            let statusColor = '#94a3b8';
            let statusText = store.status;
            
            if (store.status === 'completed') {
                borderStyle = '2px solid #10b981';
                statusColor = '#10b981';
                statusText = 'Completed ✓';
            } else if (store.status === 'navigating') {
                borderStyle = '2px solid #3b82f6';
                statusColor = '#60a5fa';
                statusText = 'Navigating... 🚀';
            } else if (store.status === 'arrived') {
                borderStyle = '2px solid #fbbf24';
                statusColor = '#fbbf24';
                statusText = 'Arrived 📍';
            } else if (store.status === 'failed') {
                borderStyle = '2px solid #ef4444';
                statusColor = '#f87171';
                statusText = 'Failed ✗';
            } else {
                statusText = 'Pending';
            }
            
            storeDiv.style.borderLeft = borderStyle;
            storeDiv.style.paddingLeft = '0.6rem';
            storeDiv.style.marginBottom = '0.4rem';
            
            // Header row
            const header = document.createElement('div');
            header.style.display = 'flex';
            header.style.justifyContent = 'space-between';
            header.style.alignItems = 'center';
            header.style.fontWeight = '600';
            
            const storeName = document.createElement('span');
            storeName.textContent = store.category;
            
            const storeStatusText = document.createElement('span');
            storeStatusText.textContent = statusText;
            storeStatusText.style.color = statusColor;
            storeStatusText.style.fontSize = '0.8rem';
            
            header.appendChild(storeName);
            header.appendChild(storeStatusText);
            storeDiv.appendChild(header);
            
            // Items sub-list
            const itemsList = document.createElement('div');
            itemsList.style.marginLeft = '0.4rem';
            itemsList.style.marginTop = '0.2rem';
            itemsList.style.display = 'flex';
            itemsList.style.flexDirection = 'column';
            itemsList.style.gap = '0.15rem';
            itemsList.style.color = '#94a3b8';
            itemsList.style.fontSize = '0.8rem';
            
            store.items.forEach(item => {
                const itemDiv = document.createElement('div');
                itemDiv.style.display = 'flex';
                itemDiv.style.alignItems = 'center';
                itemDiv.style.gap = '0.4rem';
                
                const dot = document.createElement('span');
                dot.style.width = '6px';
                dot.style.height = '6px';
                dot.style.borderRadius = '50%';
                
                const itemName = document.createElement('span');
                
                if (item.status === 'completed') {
                    dot.style.background = '#10b981';
                    itemName.textContent = `${item.name} (Picked up)`;
                    itemName.style.textDecoration = 'line-through';
                    itemName.style.color = '#475569';
                } else if (item.status === 'picking_up') {
                    dot.style.background = '#fbbf24';
                    itemName.textContent = `${item.name} (Picking up...)`;
                    itemName.style.color = '#fbbf24';
                } else {
                    dot.style.background = '#64748b';
                    itemName.textContent = `${item.name} (Pending)`;
                }
                
                itemDiv.appendChild(dot);
                itemDiv.appendChild(itemName);
                itemsList.appendChild(itemDiv);
            });
            
            storeDiv.appendChild(itemsList);
            todoContainer.appendChild(storeDiv);
        });
        
        // Add final pickup destination at the end of the list if tasks exist
        if (todoList.stores.length > 0) {
            const finalDiv = document.createElement('div');
            let borderStyle = '2px solid #334155';
            let statusColor = '#94a3b8';
            let statusText = 'Pending';
            
            if (todoList.status === 'completed') {
                borderStyle = '2px solid #10b981';
                statusColor = '#10b981';
                statusText = 'Completed ✓';
            } else if (todoList.stores.every(s => s.status === 'completed' || s.status === 'failed')) {
                borderStyle = '2px solid #3b82f6';
                statusColor = '#60a5fa';
                statusText = 'Navigating... 🚀';
            }
            
            finalDiv.style.borderLeft = borderStyle;
            finalDiv.style.paddingLeft = '0.6rem';
            finalDiv.style.marginTop = '0.4rem';
            
            const header = document.createElement('div');
            header.style.display = 'flex';
            header.style.justifyContent = 'space-between';
            header.style.alignItems = 'center';
            header.style.fontWeight = '600';
            
            const name = document.createElement('span');
            name.textContent = 'Pickup Point';
            
            const status = document.createElement('span');
            status.textContent = statusText;
            status.style.color = statusColor;
            status.style.fontSize = '0.8rem';
            
            header.appendChild(name);
            header.appendChild(status);
            finalDiv.appendChild(header);
            todoContainer.appendChild(finalDiv);
        }
    };

    const updateTelemetry = async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (robotPoseEl && data.x !== null) {
                robotPoseEl.textContent = `[X: ${data.x}, Y: ${data.y}, θ: ${data.yaw}]`;
            }
            if (shopsCountEl) {
                shopsCountEl.textContent = `${data.shops_detected}`;
            }
            if (mapCoverageEl) {
                mapCoverageEl.textContent = `${data.explored_area} m²`;
            }
            if (tasksCountEl) {
                tasksCountEl.textContent = `${data.tasks_fulfilled} tasks`;
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

            // Chat interface visibility logic
            const chatInterface = document.getElementById('chat-interface-section');
            const aiToggleBtn = document.getElementById('ai-toggle');
            if (chatInterface && aiToggleBtn) {
                if (aiToggleBtn.checked) { // Remote AI
                    if (data.has_api_key) {
                        chatInterface.style.display = 'flex';
                    } else {
                        chatInterface.style.display = 'none';
                    }
                } else { // Local AI
                    chatInterface.style.display = 'flex';
                }
            }

            // Update select-mapped-shop
            const selectMappedShop = document.getElementById('select-mapped-shop');
            if (selectMappedShop && data.mapped_shops) {
                // only update if currently not focused
                if (document.activeElement !== selectMappedShop) {
                    const currentVal = selectMappedShop.value;
                    let optionsHtml = '<option value="" disabled selected>Select a Mapped Shop...</option>';
                    data.mapped_shops.forEach(s => {
                        optionsHtml += `<option value="${s.name}">${s.name} (${s.type})</option>`;
                    });
                    selectMappedShop.innerHTML = optionsHtml;
                    if (currentVal && data.mapped_shops.find(s => s.name === currentVal)) {
                        selectMappedShop.value = currentVal;
                    }
                }
            }

            // Dynamic Planner Select UI updates
            const plannerSelect = document.getElementById('planner-select');
            if (plannerSelect && data.local_planner && document.activeElement !== plannerSelect) {
                plannerSelect.value = data.local_planner;
            }

            // Dynamic Overshoot UI updates
            const overshootValEl = document.getElementById('overshoot-val');
            if (overshootValEl && data.overshoot_cm !== undefined && document.activeElement !== overshootValEl) {
                overshootValEl.value = data.overshoot_cm;
            }

            // Sync Chat history
            if (data.chat_messages) {
                syncChat(data.chat_messages);
            }

            // Update active todo list
            if (data.todo_list) {
                updateTodoList(data.todo_list);
            } else {
                updateTodoList(null);
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
                    logRosConsole(log);
                });
            }
        } catch (error) {
            console.error("Failed to fetch logs", error);
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

    // Poll status every 500ms
    setInterval(updateTelemetry, 500);
    // Poll logs every 500ms
    setInterval(updateLogs, 500);
});
