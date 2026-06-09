# Mobile Delivery Robot (MDR)

An autonomous navigation and manipulation robot stack built on ROS. The system integrates real-time SLAM, AprilTag-based global frame correction, dynamic costmap generation, local trajectory planning (TEB), YOLO-based target localization, inverse kinematics (IK) manipulation, and high-level OpenAI LLM task orchestration. All controls are exposed through a custom, interactive Web Dashboard UI.

---

## 🌟 Key Features

*   **Central Web Dashboard UI**: Flask-based web interface offering manual D-Pad control, real-time SLAM and local planner visualizers, live camera streams, taxi navigation mapping, and chatbot controls.
*   **OpenAI LLM Task Orchestration**: Accepts natural language chat prompts (e.g. *"Bring me coffee from the Cafe and medicine from the Pharmacy"*), utilizes the `gpt_llm_client` to compile them into a structured shopping list of stores and items, and schedules the sequential navigation goals.
*   **Dynamic Process Optimization**: To save compute on embedded hardware, navigation planners and local costmaps are launched dynamically by the coordinator node only during travel, and terminated when the robot is idle or performing manipulation.
*   **YOLO Storefront Approach**: When in Delivery Mode, once the robot arrives in the vicinity of a target store, it executes a stepped visual sweep using the camera, runs YOLO inference to locate the storefront signboard, centers itself on it, uses depth raycasting to calculate exact counter distance, and plans a perpendicular alignment path.
*   **Target Coordinate Calibration & Visual Servoing**: Employs a trained YOLO model (`best.engine`) and 3D camera depth to calculate item positions. Applies a linear regression calibration matrix to map targets into arm coordinate frames for IK-guided physical picks.
*   **Chassis-Arm Grasping Pipeline**: Features a visual servoing feedback loop that aligns the chassis laterally and longitudinally to a target item, creeps forward, executes a scoop nudge, grasps the item, and deposits it into the onboard storage tray.
*   **TF Frame Alignment**: Features a rolling average filter and dead-zone publisher to align SLAM maps to global landmarks using AprilTags without creating discontinuous frame jumps.

---

## ⚠️ Non-Functional / Unsupported Features

The following experimental features have been deactivated or are not supported in production:
*   **Exploration Wandering & Signboard Reading**: Dynamic exploration of unmapped mazes by reading direction arrows on physical signboards is non-functional. The coordinates of all target stores must be pre-mapped or predefined.
*   **Storefront Categorization**: Automatic classification of storefronts (either via HSV color segmentation or VLM image calls) is unreliable. The store types must be pre-registered in the database.

---

## 🏗️ Repository Architecture

The repository is structured as a Catkin workspace:

```
MDR_Project/
├── README.md                           # Repository Documentation (This file)
├── catkin_ws/
│   └── src/
│       ├── robot_web_ui/               # Flask server, dashboard HTML/CSS/JS assets
│       ├── controller/                 # YOLO detector node and pick-and-place arm controllers
│       ├── hw4_exploration/            # Main launchers and TF mapping
│       ├── control_space_planner/      # TEB planner config and vector planning wrappers
│       ├── local_costmap_generator/    # Point cloud heightmap and obstacle costmap nodes
│       ├── gpt_llm_client/             # OpenAI client interface for task list parsing
│       ├── AprilTagLocalization/       # Landmark tag configuration maps and nodes
│       └── nexus_4wd_mecanum_simulator # Gazebo robot simulation environments
```

---

## ⚙️ Dependencies & Prerequisites

Before running the stack, ensure the following dependencies are installed:

### ROS Packages
*   ROS Melodic or Noetic (Desktop-Full recommended)
*   `rtabmap_ros`
*   `apriltag_ros`
*   `web_video_server`
*   `depth_image_proc`

### Python Libraries
```bash
pip3 install Flask flask-socketio ultralytics openai opencv-python numpy torch
```

---

## 🚀 Quick Start Instructions

### 1. Build the Catkin Workspace
Clone this repository and compile the workspace:
```bash
cd MDR_Project/catkin_ws
catkin_make
source devel/setup.bash
```

### 2. Configure Credentials
Create a configuration file in `HW4/ChatGPT_API_KEY.txt` containing your OpenAI API key, or export it in your terminal environment:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
*Alternatively, you can supply your API key at runtime directly through the dashboard UI.*

### 3. Launching the Robot Stack
The entry point of the entire application is the unified launch file `run_all.launch` in the `hw4_exploration` package.

#### Running in Simulation:
To launch the Gazebo simulation environment, SLAM, perception nodes, and the dashboard server:
```bash
roslaunch hw4_exploration run_all.launch sim:=true
```

#### Running on the Real Robot:
To launch the real camera drivers, hardware publishers, SLAM, and the dashboard:
```bash
roslaunch hw4_exploration run_all.launch sim:=false
```

### 4. Open the Web Dashboard
Open your web browser and navigate to:
```
http://localhost:5000/
```

---

## 🧭 Core Workflow Guides

### 🚕 Taxi Mode (Map-Based Navigation)
1.  Navigate to the **Taxi** tab on the Dashboard.
2.  Type in absolute `[X, Y]` coordinates in the target boxes and click **Go to Coordinates**, or
3.  **Single-click** the SLAM map image to select a location, then **double-click** to command the robot to plan a path and navigate there.

### 📦 Delivery Mode (LLM-Driven Autonomous Shopping)
1.  Navigate to the **Delivery** tab on the Dashboard.
2.  Input your OpenAI API key in the session field (if not configured via files).
3.  Enter your natural language instruction in the Chatbot input area (e.g. *"Bring me coffee from the Cafe and medicine from the Pharmacy"*).
4.  The system will:
    *   Query OpenAI via `gpt_llm_client` to compile a structured shopping list.
    *   Resolve targets to pre-mapped store coordinates or landmark tag IDs.
    *   Dynamically launch local costmaps and TEB planners to navigate to the approach coordinate.
    *   **YOLO Storefront Approach**: Initiate a stepped visual sweep, run YOLO to detect the storefront signboard, visually center the camera on the signboard, calculate exact counter distance using depth raycasting, and plan a perpendicular alignment path.
    *   Arrive at the storefront counter, stop navigation, and enable the YOLO Grab inference model.
    *   Grasp the item via visual servoing, store it in the tray, verify the pick, and proceed to the next target.
    *   Return to the starting/delivery point once all items are gathered.

---

## 📐 Math & Transformation Summary

### Coordinate Chain
The coordinate transformations flow as:
$$\text{map} \xrightarrow{\text{map\_odom\_publisher}} \text{rtabmap\_map} \xrightarrow{\text{rtabmap}} \text{odom} \xrightarrow{\text{wheel/visual odometry}} \text{base\_footprint}$$

### YOLO Target to Arm Coordinates Mapping
Target objects are extracted from pixel depth maps and mapped to the physical arm workspace using a calibrated linear regression model:
$$\text{arm\_x} = 1.445 \cdot X_{\text{cam}} - 0.350 \cdot Z_{\text{cam}} + 0.035$$
$$\text{arm\_y} = 0.015 \cdot X_{\text{cam}} - 0.316 \cdot Z_{\text{cam}} + 0.299$$
$$\text{arm\_z} = -0.015\text{ m (Fixed Grasping Height)}$$

---

## 🔒 License
This project is licensed under the Modified BSD Software License.
