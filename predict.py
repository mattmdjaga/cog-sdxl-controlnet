import os
import sys
import json
import shutil
from typing import List, Tuple, Dict
from itertools import combinations_with_replacement, product
from collections import defaultdict
import cv2
import torch
import numpy as np
from PIL import Image
from transparent_background import Remover
from RealESRGAN import RealESRGAN
from cog import BasePredictor, Input, Path
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
    AutoProcessor,
    CLIPModel,
)
import huggingface_hub

from roop_custom.utils import (
    get_codeformer_models,
    encode_execution_providers,
    decode_execution_providers,
    suggest_execution_providers,
    suggest_execution_threads,
    inference_codeformer,
)

from roop_custom import roop
from roop_custom.roop.utilities import normalize_output_path
from roop_custom.roop.processors.frame.core import get_frame_processors_modules
import concurrent.futures

roop.globals.many_faces = False
roop.globals.reference_face_position = 0
roop.globals.reference_frame_number = 0
roop.globals.similar_face_distance = 0.85
roop.globals.execution_providers = decode_execution_providers(["cpu"])
roop.globals.execution_threads = suggest_execution_threads()
roop.globals.headless = True

CONTROL_MODEL = "diffusers/controlnet-canny-sdxl-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
MODEL_NAME = "nyxia/endjourney-xl"
LORA_NAME = "mattmdjaga/starbucks_lora"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
MODEL_CACHE = "sdxl-cache"
LORA_CACHE = "lora-cache"

MASKFORMER_CACHE = "maskformer"
SEGFORMER_CACHE = "segformer"
REALESRGAN_CACHE = "realesrgan"
CLIP_CACHE_LARGE = "clip-vit-large-patch14"
CLIP_CACHE_BASE = "clip-vit-base-patch32"

MASKFORMER_MODEL = "facebook/mask2former-swin-large-coco-panoptic"
SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
CLIP_MODEL_LARGE = "openai/clip-vit-large-patch14"
CLIP_MODEL_BASE = "openai/clip-vit-base-patch32"

keys = json.load(open("keys.json"))
huggingface_hub.login(keys["hf"])


def get_ids(result: Dict) -> List[int]:
    """
    Extracts ids of people and animals from the panoptic segmentation result.
    Args:
        result: panoptic segmentation result
    Returns:
        list of ids of people
        list of ids of animals
    """
    ids = []
    animal_ids = []
    for segment_info in result["segments_info"]:
        if segment_info["label_id"] == 0:
            ids.append(segment_info["id"])
        elif segment_info["label_id"] in [15, 16]:
            animal_ids.append(segment_info["id"])
    return ids, animal_ids


def find_order_of_ids(ids: List[int], mask: torch.Tensor) -> List[int]:
    """
    Finds the order of people in the image from left to right
    Args:
        ids: list of ids of people
        mask: panoptic segmentation mask
    Returns:
        list of ids of people in the image from left to right
    """
    ordered_ids = []
    column = 0
    positions = defaultdict(list)
    # find the left most position of each id
    while ids:
        column += 1
        column_mask = mask[:, column]
        for tag in ids:
            if (column_mask == tag).any():
                ids.remove(tag)
                positions[tag].append(column)
    # find the right most position of each id
    column = mask.shape[1]
    ids = list(positions.keys())
    while ids:
        column -= 1
        column_mask = mask[:, column]
        for tag in ids:
            if (column_mask == tag).any():
                ids.remove(tag)
                positions[tag].append(column)

    # return the ordered ids based on the average of the left and right most positions
    for tag in positions:
        ordered_ids.append((tag, sum(positions[tag]) / 2))
    ordered_ids.sort(key=lambda x: x[1])
    ordered_ids = [tag for tag, _ in ordered_ids]
    return ordered_ids


def resize_and_crop(image, desired_width=1356, desired_height=840):
    """
    Resize and crop an image to fit the specified dimensions.
    Args:
        image: PIL image
        desired_width: desired width
        desired_height: desired height
    Returns:
        resized and cropped PIL image
    """
    target_ratio = desired_width / desired_height
    img_ratio = image.width / image.height

    if img_ratio > target_ratio:
        new_width = int(target_ratio * image.height)
        new_height = image.height
    else:
        new_width = image.width
        new_height = int(image.width / target_ratio)

    left = (image.width - new_width) / 2
    top = (image.height - new_height) / 2
    right = (image.width + new_width) / 2
    bottom = (image.height + new_height) / 2

    image = image.crop((left, top, right, bottom))
    image = image.resize((desired_width, desired_height), Image.LANCZOS)

    return image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE, torch_dtype=torch.float16
        )
        better_vae = AutoencoderKL.from_pretrained(VAE_CACHE, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            MODEL_CACHE,
            vae=better_vae,
            controlnet=controlnet,
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        self.pipe = pipe.to("cuda")
        pipe.load_lora_weights(LORA_NAME)
        pipe.fuse_lora(lora_scale=1.0)

        upsampler, codeformer_net, device = get_codeformer_models()
        self.upsampler = upsampler
        self.codeformer_net = codeformer_net
        self.device = device

        self.processor_maskformer = AutoImageProcessor.from_pretrained(
            MASKFORMER_CACHE, torch_dtype=torch.float16
        )
        self.model_maskformer = Mask2FormerForUniversalSegmentation.from_pretrained(
            MASKFORMER_CACHE, torch_dtype=torch.float16
        ).to(device)
        clip_names = [CLIP_CACHE_BASE, CLIP_CACHE_LARGE]
        self.processor_segformer = SegformerImageProcessor.from_pretrained(
            SEGFORMER_CACHE, torch_dtype=torch.float16
        )
        self.model_segformer = AutoModelForSemanticSegmentation.from_pretrained(
            SEGFORMER_CACHE, torch_dtype=torch.float16
        ).to(device)
        clip_models = []
        clip_processors = []
        for clip_name in clip_names:
            clip_processors.append(AutoProcessor.from_pretrained(clip_name))
            clip_models.append(CLIPModel.from_pretrained(clip_name).to("cuda"))

        self.clip_processors = clip_processors
        self.clip_models = clip_models
        races = ["white", "asian", "black", "indian"]
        genders = [
            "family",
            "baby",
            "male adult",
            "female adult",
            "male friends",
            "female friends",
            "male and female friends",
            "male child",
            "male teenager",
            "female child",
            "female teenager",
            "couple",
            "freinds",
        ]
        race_gender_combinations = [
            f"{race} {gender}" for race, gender in product(races, genders)
        ]
        self.people = self.generate_people_combinations(race_gender_combinations, 1)

        self.styles = json.load(open("styles.json"))

        self.model2 = RealESRGAN(device, scale=2)
        self.model2.load_weights("upscaler/RealESRGAN_x2.pth", download=True)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def img2img_cany(
        self,
        img,
        prompt,
        negative_prompt,
        num_inference_steps,
        condition_scale,
        strength,
    ):  
        image = np.array(img)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        print(f"Processed img: {type(image)}")
        images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=img,
            control_image=image,
            controlnet_conditioning_scale=condition_scale,
            num_inference_steps=num_inference_steps,
            strength=strength,
        ).images
        return images

    def get_seg_info(self, img: Image.Image) -> Tuple[Dict, List[int]]:
        """
        get segmentation info
        Args:
            img: image

        Returns:
            result: result
            ordered_ids: ordered ids
            animal_ids: animal ids
        """
        inputs = self.processor_maskformer(images=img, return_tensors="pt").to("cuda")
        inputs_half = {k: v.half() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model_maskformer(**inputs_half)
        result = self.processor_maskformer.post_process_panoptic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0]
        ids, animal_ids = get_ids(result)
        ordered_ids = find_order_of_ids(ids, result["segmentation"])
        return result, ordered_ids, animal_ids

    @staticmethod
    def generate_people_combinations(people: List, max_group_size: int) -> List[str]:
        """
        Generates all possible combinations of people.
        Args:
            people: list of people
            max_group_size: maximum group size
        Returns:
            list of combinations
        """
        all_combinations = set()  # Using a set to avoid duplicates
        for group_size in range(1, max_group_size + 1):
            for group in combinations_with_replacement(people, group_size):
                all_combinations.add(" and ".join(group))
        return sorted(all_combinations)

    def get_identiy(self, image: Image.Image, top_k: int = 5) -> str:
        """
        Gets the identity of the person in the image.
        Args:
            models: CLIP models
            processors: CLIP processors
            people: list of people
            img_file: image file
            top_k: top k predictions
        Returns:
            identity of the person in the image
        """
        scores = defaultdict(list)
        for processor, model in zip(self.clip_processors, self.clip_models):
            inputs = processor(
                text=self.people, images=image, return_tensors="pt", padding=True
            ).to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].half()
            outputs = model(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)
            values, predictions = probs.topk(top_k)
            # Print the result
            for i in range(top_k):
                scores[self.people[predictions[0][i]]].append(values[0][i].item())
        # return the top identity
        return max(scores, key=lambda x: sum(scores[x]))

    def process_animal_mask(self, result, img_base, animal_ids, data, style_type, h, w):
        temp_animal_mask = result["segmentation"].cpu()
        for id in animal_ids:
            temp_animal_mask[temp_animal_mask == id] = 254

        animal_mask = np.array(img_base)
        animal_mask[temp_animal_mask != 254] = 130
        animal_img = Image.fromarray(animal_mask)
        tags = [r["label_id"] for r in result["segments_info"]]

        if 15 in tags and 16 in tags:
            who = "a cat and a dog"
        elif 15 in tags:
            who = "a cat"
        else:
            who = "a dog"

        prompt = data[style_type]["prompt"].replace("{prompt}", who)
        negative_prompt = data[style_type]["negative_prompt"]

        animals_img = self.img2img_cany(
            animal_img, prompt, negative_prompt, 20, condition_scale=0.2, strength=0.5
        )[0]

        return temp_animal_mask, animals_img

    @staticmethod
    def removebg_inpy(image):
        remover = Remover(device="cuda")
        image = image.convert("RGB")
        out = remover.process(image)
        # image = Image.fromarray(out)
        image = out
        out = None  # Release the output variable
        torch.cuda.empty_cache()  # Clear GPU memory cache
        return image

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        style_type: str = Input(
            description="The style to use for this generation",
            default="1",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=20
        ),
        condition_scale: float = Input(
            description="controlnet conditioning scale for generalization",
            default=0.1,
            ge=0.0,
            le=1.0,
        ),
        strength: float = Input(
            description="Strength of noise in init image",
            default=0.75,
            ge=0.0,
            le=1.0,
        ),
        width: int = Input(
            description="Width of the output image", default=1024, ge=1, le=4096
        ),
        height: int = Input(
            description="Height of the output image", default=1024, ge=1, le=4096
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        image = self.load_image(image)
        w, h = width, height
        img_base = resize_and_crop(image, desired_width=w, desired_height=h)
        src_image = img_base.copy()

        result, ordered_ids, animal_ids = self.get_seg_info(img_base)
        if len(ordered_ids) > 4:
            print(f"Too many people in the image: {len(ordered_ids)}")
            return img_base
        temp_mask = result["segmentation"].cpu()
        for id in ordered_ids:
            temp_mask[temp_mask == id] = 254
        mask = np.array(img_base)
        mask[temp_mask != 254] = 130
        img = Image.fromarray(mask)
        who = self.get_identiy(img)
        #who = "asian man"
        prompt = self.styles[style_type]["prompt"].replace("{prompt}", who)
        negative_prompt = self.styles[style_type]["negative_prompt"]

        images = self.img2img_cany(
            img, prompt, negative_prompt, num_inference_steps, condition_scale, strength
        )

        args = [[src_image, images[0]]]
        gen_imgs = []
        frame_processors = ["face_swapper"]
        for frame_processor in get_frame_processors_modules(frame_processors):
            for arg in args:
                gen_arr = frame_processor.process_image(arg)
                gen_img = cv2.cvtColor(gen_arr, cv2.COLOR_BGR2RGB)
                gen_imgs.append(gen_img)

        args_codeformer = [
            (
                np.array(img),
                True,
                False,
                False,
                0,
                1.0,
                self.upsampler,
                self.codeformer_net,
                self.device,
            )
            for img in gen_imgs
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            codeformer_res = list(executor.map(inference_codeformer, args_codeformer))
        codeformer_res = [Image.fromarray(result) for result in codeformer_res]
        codeformer_res[0] = codeformer_res[0].resize((w, h))
        img2img2 = self.img2img_cany(
            codeformer_res[0],
            prompt,
            negative_prompt,
            num_inference_steps,
            condition_scale=0.8,
            strength=0.05,
        )

        if len(animal_ids) != 0:
            temp_animal_mask, animals_img = self.process_animal_mask(
                result, img_base, animal_ids, self.styles, style_type, h, w
            )

            se_image = img2img2[0].resize((w, h))
            se_arr = np.array(se_image)
            animals_arr = np.array(animals_img)
            se_arr[temp_animal_mask == 254] = animals_arr[temp_animal_mask == 254]

            se_image = Image.fromarray(se_arr)
        else:
            se_image = img2img2[0]

        big_img = self.model2.predict(se_image)
        final_img = self.removebg_inpy(big_img)

        output_path = f"/tmp/output.png"
        final_img.save(output_path)

        return Path(output_path)
