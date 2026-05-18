document.addEventListener('DOMContentLoaded', () => {
    // Layout Slider Logic
    const layoutSlider = document.getElementById('layout-slider');
    const layoutVal = document.getElementById('layout-val');
    
    const updateLayout = (widthPercent) => {
        document.documentElement.style.setProperty('--col-left', `${widthPercent}%`);
    };
    
    if (layoutSlider && layoutVal) {
        // Load initial layout preference if saved in localStorage
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

    // Mode Toggle Logic (Placeholder for now)
    const modeToggle = document.getElementById('mode-toggle');
    const labelRemote = document.getElementById('label-remote');
    const labelExplore = document.getElementById('label-explore');

    // Currently exploration is disabled, so we force the toggle back if clicked
    modeToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            e.preventDefault();
            setTimeout(() => {
                modeToggle.checked = false;
                alert("Exploration mode is currently under development.");
            }, 200);
        }
    });

    // Remote Control Logic
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
            console.error("Failed to send command", error);
            // Visual feedback for error could be added here
        }
    };

    const startCommand = (linear, angular) => {
        if (activeInterval) clearInterval(activeInterval);
        currentCmd = { linear, angular };
        sendCommand(linear, angular);
        // Continuously send command while holding (10Hz)
        activeInterval = setInterval(() => sendCommand(linear, angular), 100);
    };

    const stopCommand = () => {
        if (activeInterval) clearInterval(activeInterval);
        currentCmd = { linear: 0, angular: 0 };
        sendCommand(0, 0);
    };

    // Attach event listeners to buttons
    Object.values(buttons).forEach(btn => {
        if (!btn) return;

        const linear = parseFloat(btn.dataset.linear);
        const angular = parseFloat(btn.dataset.angular);

        // Mouse events
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

        // Touch events for mobile support
        btn.addEventListener('touchstart', (e) => {
            e.preventDefault(); // Prevent scrolling
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
        if (e.repeat) return; // Prevent key repeat triggering multiple starts
        
        switch(e.key) {
            case 'ArrowUp':
            case 'w':
            case 'W':
                startCommand(0.5, 0);
                buttons['btn-up'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowDown':
            case 's':
            case 'S':
                startCommand(-0.5, 0);
                buttons['btn-down'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowLeft':
            case 'a':
            case 'A':
                startCommand(0, 1.0);
                buttons['btn-left'].style.transform = 'translateY(2px)';
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                startCommand(0, -1.0);
                buttons['btn-right'].style.transform = 'translateY(2px)';
                break;
            case ' ': // Spacebar for stop
                stopCommand();
                buttons['btn-stop'].style.transform = 'translateY(2px)';
                break;
        }
    });

    document.addEventListener('keyup', (e) => {
        switch(e.key) {
            case 'ArrowUp':
            case 'w':
            case 'W':
                stopCommand();
                buttons['btn-up'].style.transform = '';
                break;
            case 'ArrowDown':
            case 's':
            case 'S':
                stopCommand();
                buttons['btn-down'].style.transform = '';
                break;
            case 'ArrowLeft':
            case 'a':
            case 'A':
                stopCommand();
                buttons['btn-left'].style.transform = '';
                break;
            case 'ArrowRight':
            case 'd':
            case 'D':
                stopCommand();
                buttons['btn-right'].style.transform = '';
                break;
            case ' ':
                buttons['btn-stop'].style.transform = '';
                break;
        }
    });

    // Telemetry Polling
    const robotPoseEl = document.getElementById('robot-pose');
    const shopsCountEl = document.getElementById('shops-count');
    const mapCoverageEl = document.getElementById('map-coverage');

    const updateTelemetry = async () => {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (robotPoseEl && data.x !== null) {
                robotPoseEl.textContent = `[X: ${data.x}, Y: ${data.y}, θ: ${data.yaw}]`;
            }
            if (shopsCountEl) {
                shopsCountEl.textContent = data.shops_detected;
            }
            if (mapCoverageEl) {
                mapCoverageEl.textContent = `${data.explored_area} m²`;
            }
        } catch (error) {
            console.error("Failed to fetch telemetry status", error);
        }
    };

    // Poll every 500ms
    setInterval(updateTelemetry, 500);
});
