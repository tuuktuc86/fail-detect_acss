import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import re

def load_images_from_folder(folder, prefix, expected_count):
    images = []
    files = sorted(os.listdir(folder))
    for fname in files:
        if fname.endswith(".png") and fname.startswith(prefix):
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((360, 240))  # (w, h) = (360, 240)
            img_np = np.array(img).transpose(2, 0, 1)  # [H, W, C] → [C, H, W]
            images.append(img_np)

    images = np.stack(images)  # [T, 3, 240, 360]
    assert images.shape[0] == expected_count, f"Image count mismatch in {folder}"
    return images

def extract_episode_number(name):
    # 예: "success_episode3_steps120" → 3
    match = re.match(r"success_episode(\d+)_steps(\d+)", name)
    if match:
        return int(match.group(1)), int(match.group(2))  # (episode_number, trajectory_length)
    return float("inf"), float("inf")  # fallback for safety

def process_directory(base_dir):
    output_dict = {}
    episode_dirs = sorted(
        [d for d in os.listdir(base_dir) if d.startswith("success_episode")],
        key=lambda name: extract_episode_number(name)
    )
    
    for ep_dir in tqdm(episode_dirs, desc="Processing episodes"):
        ep_path = os.path.join(base_dir, ep_dir)
        front_path = os.path.join(ep_path, "front_view")
        top_path = os.path.join(ep_path, "top_view")
        wrist_path = os.path.join(ep_path, "wrist_view")
        robot_state_path = os.path.join(ep_path, "robot_state.npz")

        # ep_name은 저장용 key → episode_3_120 같은 형식
        ep_num, traj_len = extract_episode_number(ep_dir)
        ep_name = f"episode_{ep_num}_{traj_len}"

        # Load robot state
        robot_state = np.load(robot_state_path)
        EE_pose = robot_state["EE_pose"]
        joint_obj_action = robot_state["obs"]
        torque = robot_state["applied_torque"]

        T = EE_pose.shape[0]

        try:
            image_front = load_images_from_folder(front_path, "front_view_", T)
            image_top = load_images_from_folder(top_path, "top_view_", T)
            image_wrist = load_images_from_folder(wrist_path, "wrist_view_", T)
        except AssertionError as e:
            print(f"{ep_name}_lengthWrong: {e}")
            return  # Stop processing immediately on mismatch

        # Verify all lengths match
        if not (image_front.shape[0] == image_top.shape[0] == image_wrist.shape[0] ==
                EE_pose.shape[0] == joint_obj_action.shape[0] == torque.shape[0]):
            print(f"{ep_name}_lengthWrong")
            return
        
        print(f"{ep_name}_added")

        output_dict[ep_name] = {
            "image_front": image_front,
            "image_top": image_top,
            "image_wrist": image_wrist,
            "robot_state": {
                "EE_pose": EE_pose,
                "joint_obj_action": joint_obj_action,
                "torque": torque
            }
        }

    return output_dict

if __name__ == "__main__":
    input_root = "/AILAB-summer-school-2025/success_data_raw"
    output_path = "/AILAB-summer-school-2025/success_raw_epi{total}.pkl"

    result = process_directory(input_root)

    if result is not None:
        total_traj = len(result)
        output_file = output_path.format(total=total_traj)
        with open(output_file, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved {total_traj} episodes to {output_file}")
