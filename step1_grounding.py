
import sys
import os
import cv2
import numpy as np
import json
import argparse
import re
import logging
from controller.fairino_arm import RobotArm
from utils.other_utils import set_random_seeds

from visions.grounding.dino_grounding import DinoGrounding
from controller.extern_camera import ExternCameraCallbacks
from languages.ram_vlm import RamVLM

from utils.image_utils import save_depth_npy, save_rgb_png, draw_label_at_mask_center

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
    parser = argparse.ArgumentParser(description="Task Decomposition and Object Grasp Demo")
    parser.add_argument("--prompt", type=str, default="clean the table", 
                        help="Text prompt describing the task")
    parser.add_argument("--output", type=str, default="./output/scene_01", help="Directory to save outputs")
    parser.add_argument("--only_grounding", action='store_true', help="Only perform grounding without vlm finde objects")
    parser.add_argument("--no_color", action='store_true', help="Disable color grounding for objects")
    return parser.parse_args()

def main():
    args = parse_args()

    promp_name = re.sub(r'\W+', '_', args.prompt.lower())

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(args.output)
    
    debug_dir = os.path.join(output_dir, promp_name)
    if os.path.exists(debug_dir):
        for file in os.listdir(debug_dir):
            file_path = os.path.join(debug_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    args_json_path = os.path.join(output_dir, "instruction.json")
    with open(args_json_path, 'w') as f:
        json.dump({"prompt": args.prompt}, f, indent=4)
    logger.info(f">>> OUTPUT DIR: {output_dir}.")

    config_file = os.path.join(cur_dir, '../controller/assets', 'fr5_SHB901.json')
    robot = RobotArm(config_file=config_file)
    robot.reset()

    vlm = RamVLM()

    RAM_category_dictionary = vlm.read_RAM_categories()
    logger.info(f"RAM --> category dictionary: {RAM_category_dictionary.keys()}")
    
    config_file = os.path.join(cur_dir, 
        './controller/assets/calibration/external_camera', 'zed_config.json')
    extern_camera = ExternCameraCallbacks['ZED'](config_file=config_file)

    config_file = os.path.join(cur_dir, './visions/grounding/assets', 'dino_grounding.json')
    vgm = DinoGrounding(config_file=config_file)


    rgb_ext, depth_ext = extern_camera.capture_image()
    rgb_ext_pth = os.path.join(output_dir, "rgb_ext.png")
    save_rgb_png(rgb_ext, rgb_ext_pth)
    save_rgb_png(extern_camera.left_image, os.path.join(output_dir, "rgb_ext_left.png"))
    save_rgb_png(extern_camera.right_image, os.path.join(output_dir, "rgb_ext_right.png"))
    save_depth_npy(depth_ext, os.path.join(output_dir, "depth_ext.npy"))

    rgb_ext = cv2.imread(rgb_ext_pth)[..., ::-1]

    if not args.only_grounding:
        res_object = vlm.infer_objects(args.prompt, rgb_ext_pth, RAM_keys=RAM_category_dictionary.keys())
        related_objects = vlm.parse_json_response(res_object)
        logger.warning(f"VLM --> finded objects: \n{res_object}")
        ram_object_json_path = os.path.join(output_dir, "vlm_objects.json")
        with open(ram_object_json_path, 'w') as f:
            json.dump(related_objects, f, indent=4)
    else:
        vlm_object_file = os.path.join(output_dir, "vlm_objects.json")
        with open(vlm_object_file, 'r') as f:
            related_objects = json.load(f)

    masks = []
    xyxys = []
    class_ids = []
    count = 0

    grounding_prompt_list = []
    for object in related_objects:
        if "other" in object['category'].lower():
            grounding_prompt = object['name']
        else:
            if not args.no_color:
                grounding_prompt = f"{object['color']} {object['category']}"
            else:
                grounding_prompt = f"{object['category']}"
                
        if grounding_prompt in grounding_prompt_list:
            logger.info(f"Grounding --> skip object '{grounding_prompt}' as it already exists in the list.")
            continue

        grounding_prompt_list.append(grounding_prompt)
        logger.info(f"Grounding --> used prompts: {grounding_prompt}")

        detections = vgm.grounding(rgb_ext, grounding_prompt)

        mask = detections.mask
        xyxy = detections.xyxy
        class_id = detections.class_id

        for i in range(mask.shape[0]):
            grounding_promp_name = re.sub(r'\W+', '_', grounding_prompt.lower())
            mask_rgb = (mask[i, ..., None] * rgb_ext).astype(np.uint8)
            mask_path = os.path.join(debug_dir, f"grounding_mask_{grounding_promp_name}_{count:02}.png")
            count += 1
            save_rgb_png(mask_rgb, mask_path)

        masks.append(mask)
        xyxys.append(xyxy)
        class_ids.append(class_id)
    masks = np.concatenate(masks, axis=0)
    xyxys = np.concatenate(xyxys, axis=0)
    class_ids = np.concatenate(class_ids, axis=0)
        
    detections_npy_path = os.path.join(output_dir, "detections.npy")
    np.save(detections_npy_path, {
        "masks": masks,
        "xyxys": xyxys,
        "class_ids": class_ids
    })  
    for i in range(masks.shape[0]):
        rgb_ext = draw_label_at_mask_center(
            rgb_ext, masks[i, ...], 
            label_text=f"{i}", 
            font_scale=0.5, 
            font_thickness=2,
            text_color=(0, 255, 0),
            bg_color=(0, 0, 0)
        )
    rgb_labelling_image_path = os.path.join(output_dir, "rgb_ext_labeling.png") 
    save_rgb_png(rgb_ext, rgb_labelling_image_path)
    logger.info(f">>> save grounding rgb image to {rgb_labelling_image_path}.")

    if not args.only_grounding:

        for i in RAM_category_dictionary.keys():
            RAM_keys = ", ".join(map(str, RAM_category_dictionary.keys()))
        res_category = vlm.infer_objects_label(
            image=rgb_labelling_image_path, object_items=related_objects
        )
        object_categories = vlm.parse_json_response(res_category)
        logger.warning(f"VLM --> object categories: \n{res_category}")

        ram_category_json_path = os.path.join(output_dir, "vlm_categories.json")
        with open(ram_category_json_path, 'w') as f:
            json.dump(object_categories, f, indent=4)  

if __name__ == "__main__":
    set_random_seeds(2) 
    main()
