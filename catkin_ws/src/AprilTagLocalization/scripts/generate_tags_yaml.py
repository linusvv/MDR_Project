#!/usr/bin/env python3
import os
import yaml
import rospkg

def main():
    rospack = rospkg.RosPack()
    apriltag_path = rospack.get_path('apriltag_localization')
    
    # 1. Load signboards.yaml
    try:
        pkg_path = rospack.get_path('robot_web_ui')
        mdr_path = os.path.dirname(os.path.dirname(os.path.dirname(pkg_path)))
        signboards_path = os.path.join(mdr_path, 'HW4', 'signboards.yaml')
    except Exception:
        signboards_path = "/home/ee478_team1/catkin_ws/src/MDR_Project/HW4/signboards.yaml"
        
    with open(signboards_path, 'r') as f:
        signboards = yaml.safe_load(f)
        
    # 2. Load ee478_n1_room113.yaml
    room_path = os.path.join(apriltag_path, 'config', '2026', 'ee478_n1_room113.yaml')
    with open(room_path, 'r') as f:
        room_config = yaml.safe_load(f)
        
    # Get signboard names and sizes
    signboard_sizes = {}
    if room_config and 'TAG_TRUE_RT' in room_config and 'TAGS' in room_config['TAG_TRUE_RT']:
        for tag_def in room_config['TAG_TRUE_RT']['TAGS']:
            name = tag_def[0]
            size = float(tag_def[1])
            signboard_sizes[name] = size
            
    # 3. Build standalone_tags and tag_bundles
    standalone_tags = []
    tag_bundles = []
    
    for board_name, tags in signboards.items():
        if not isinstance(tags, dict):
            continue
            
        size = signboard_sizes.get(board_name, 0.02)
        
        layout = []
        for tag_str, data in tags.items():
            try:
                tag_id = int(tag_str)
            except ValueError:
                continue
                
            dir_str = data.get('direction', 'Up').strip().lower()
            if dir_str == 'left':
                x_offset = -0.0655
            elif dir_str == 'right':
                x_offset = 0.0879
            else:
                x_offset = 0.0112
                
            layout.append({
                'id': tag_id,
                'size': size,
                'x': x_offset,
                'y': 0.0,
                'z': 0.0,
                'qw': 1.0,
                'qx': 0.0,
                'qy': 0.0,
                'qz': 0.0
            })
            
        if layout:
            tag_bundles.append({
                'name': board_name,
                'layout': layout
            })
            
    # Write output structure
    output_config = {
        'standalone_tags': standalone_tags,
        'tag_bundles': tag_bundles
    }
    
    out_path = os.path.join(apriltag_path, 'config', '2026', 'generated_tags.yaml')
    with open(out_path, 'w') as f:
        yaml.safe_dump(output_config, f, default_flow_style=False)
        
    print(f"Generated tags config at {out_path}")

if __name__ == '__main__':
    main()
