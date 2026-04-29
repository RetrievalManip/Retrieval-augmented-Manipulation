import sys
import os 
import numpy as np
from os.path import join as pjoin
import json
import argparse
import re
import logging

from languages.ram_vlm import RamVLM

from utils.other_utils import set_random_seeds

sys.stdout.reconfigure(encoding='utf-8')
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[94m',
        'DEBUG': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger()
logger.handlers[0].setFormatter(ColoredFormatter('%(levelname)s: %(message)s'))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VLM action constraints from RAM primitives")
    parser.add_argument("--prompt", type=str, default="put the bowl into the plate", help="Text prompt describing the task")
    parser.add_argument("--output", type=str, default="./output/scene_01", help="Directory to save outputs")
    return parser.parse_args()


def save_json(data, output_dir, debug_dir, file_name):
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    debug_path = os.path.join(debug_dir, file_name)
    with open(debug_path, 'w') as f:
        json.dump(data, f, indent=4)


def require_json_list(data, stage_name):
    if not isinstance(data, list):
        raise ValueError(f"{stage_name} should return a JSON list, got {type(data).__name__}.")
    return data


def build_subtask_instruction(task_prompt, subtask, subtask_index):
    if isinstance(subtask, dict):
        subtask_name = subtask.get("subtask", f"Subtask {subtask_index}")
        subtask_goal = subtask.get("goal", subtask.get("subtask_goal", ""))
        return (
            f"Original task: {task_prompt}\n"
            f"Current subtask: {subtask_name}\n"
            f"Subtask goal: {subtask_goal}"
        )

    return (
        f"Original task: {task_prompt}\n"
        f"Current subtask: {subtask}"
    )


def normalize_action_output(action_data, first_action_id):
    if isinstance(action_data, dict):
        action_items = [action_data]
    elif isinstance(action_data, list):
        action_items = action_data
    else:
        raise ValueError(
            f"Action decomposition should return a JSON object or list, got {type(action_data).__name__}."
        )

    normalized_actions = []
    for index, action in enumerate(action_items):
        if not isinstance(action, dict):
            raise ValueError(
                f"Each action decomposition item should be a JSON object, got {type(action).__name__}."
            )
        action["subtask_id"] = first_action_id + index
        if "action_type" not in action and "action" in action:
            action["action_type"] = action["action"]
        normalized_actions.append(action)

    return normalized_actions


def build_ram_object_description(vlm_categories, ram_objects_meta, ram_category_dictionary):
    objects_dict = {
        str(item['id']): {
            'name': item['name'],
            'category': item['category'],
        }
        for item in vlm_categories
    }

    objects_description_list = []
    for object_id, meta in ram_objects_meta.items():
        grasp_primitives = [k for k in meta if k.startswith("grasp_")]
        plane_primitives = [k for k in meta if k.startswith("plane_")]

        object_meta = objects_dict[str(object_id)]
        object_name = object_meta['name']
        object_category = object_meta['category']

        if object_category == "other":
            summary = f"- For {object_name} with label {object_id} belonging to category '{object_category}', we have: "
        else:
            object_length, object_width, object_height = meta['object_size'] / 10.0
            summary = (
                f"- For {object_name} with label {object_id} belonging to category '{object_category}', "
                f"the length, width and height of this object are {object_length:.2f} cm, "
                f"{object_width:.2f} cm and {object_height:.2f} cm respectively. "
            )

        if grasp_primitives and plane_primitives:
            summary += (
                f"We define {len(grasp_primitives)} grasp point(s) and "
                f"{len(plane_primitives)} plane(s) with the object centroid as follows: "
            )
        elif plane_primitives:
            summary += f"We define {len(plane_primitives)} plane(s) and the object centroid as follows: "
        elif grasp_primitives:
            summary += f"We define {len(grasp_primitives)} grasp point(s) and the object centroid as follows: "

        object_description = summary
        category_primitives = ram_category_dictionary.get(object_category, {}).get('primitives', {})
        for primitive_name in grasp_primitives + plane_primitives:
            primitive_meta = category_primitives.get(primitive_name)
            if primitive_meta is None:
                continue
            object_description += f"\n    - '{primitive_name}': '{primitive_meta['description']}'"
        object_description += "\n    - 'centroid': 'An object centroid is the geometric center of a shape or object, representing the arithmetic mean position of all its points'"

        objects_description_list.append(object_description)

    return "\n".join(objects_description_list)


if __name__ == "__main__":
    set_random_seeds(42)
    args = parse_args()

    promp_name = re.sub(r'\W+', '_', args.prompt.lower())
    logger.info(f"Prompt name: {promp_name}")

    output_dir = os.path.join(args.output)
    debug_dir = os.path.join(args.output, f"{promp_name}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    rgb_labeling_image_path = os.path.join(output_dir, "rgb_ext_labeling.png")

    vlm_category_json = pjoin(output_dir, 'vlm_categories.json')
    with open(vlm_category_json, 'r') as f:
        vlm_categories = json.load(f)

    ram_objects_meta_file = pjoin(output_dir, 'ram_objects_meta.npy')
    with open(ram_objects_meta_file, 'rb') as f:
        ram_objects_meta = np.load(f, allow_pickle=True).item()
    for object_id, meta in ram_objects_meta.items():
        logger.info(f"Object ID: {object_id}, primitives: {meta.keys()}")

    vlm = RamVLM()
    ram_category_dictionary = vlm.read_RAM_categories()
    ram_object_desc = build_ram_object_description(
        vlm_categories,
        ram_objects_meta,
        ram_category_dictionary,
    )
    logger.info(f"RAM --> objects description: \n{ram_object_desc}")

    res_subtasks = vlm.decomp_longtask2subtask(
        args.prompt,
        rgb_labeling_image_path,
        RAM_object=ram_object_desc,
        max_tokens=10000,
    )
    logger.warning(f"VLM --> decomposed tasks: \n{res_subtasks}")
    vlm_subtasks = require_json_list(vlm.parse_json_response(res_subtasks), "Long-task decomposition")
    save_json(vlm_subtasks, output_dir, debug_dir, "vlm_subtasks.json")

    vlm_actions = []
    for subtask_index, subtask in enumerate(vlm_subtasks, start=1):
        subtask_instruction = build_subtask_instruction(args.prompt, subtask, subtask_index)
        res_actions = vlm.decomp_subtask2action(
            subtask_instruction,
            rgb_labeling_image_path,
            RAM_object=ram_object_desc,
            max_tokens=10000,
        )
        logger.warning(f"VLM --> action constraints for subtask {subtask_index}: \n{res_actions}")
        action_data = vlm.parse_json_response(res_actions)
        vlm_actions.extend(normalize_action_output(action_data, len(vlm_actions) + 1))

    save_json(vlm_actions, output_dir, debug_dir, "vlm_actions.json")
    logger.info(f"Saved {len(vlm_actions)} VLM action constraint(s) to {os.path.join(output_dir, 'vlm_actions.json')}.")
