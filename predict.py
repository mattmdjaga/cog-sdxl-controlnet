from cog import BasePredictor, Input, Path
import os
import sys
import cv2
import time
import torch
import shutil
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image
import huggingface_hub

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#current_dir = os.path.dirname(os.path.abspath(__file__))
#print(f"Current script directory: {current_dir}")
#print(os.listdir(current_dir))
#print(os.listdir("/roop_custom/CodeFormer/CodeFormer/weights"))
sys.path.append("/roop_custom/CodeFormer/CodeFormer")

from basicsr.utils.download_util import load_file_from_url

pretrain_model_url = {
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "detection": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
    "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}
# download weights
if not os.path.exists("/roop_custom/CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"):
    load_file_from_url(
        url=pretrain_model_url["codeformer"],
        model_dir="/roop_custom/CodeFormer/CodeFormer/weights/CodeFormer",
        progress=True,
        file_name=None,
    )
if not os.path.exists(
    "/roop_custom/CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth"
):
    load_file_from_url(
        url=pretrain_model_url["detection"],
        model_dir="/roop_custom/CodeFormer/CodeFormer/weights/facelib",
        progress=True,
        file_name=None,
    )
if not os.path.exists("/roop_custom/CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth"):
    load_file_from_url(
        url=pretrain_model_url["parsing"],
        model_dir="/roop_custom/CodeFormer/CodeFormer/weights/facelib",
        progress=True,
        file_name=None,
    )
if not os.path.exists("/roop_custom/CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth"):
    load_file_from_url(
        url=pretrain_model_url["realesrgan"],
        model_dir="/roop_custom/CodeFormer/CodeFormer/weights/realesrgan",
        progress=True,
        file_name=None,
    )

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

os.environ['HUGGINGFACE_TOKEN'] = "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"
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
huggingface_hub.login(huggingface_token)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        print("Loading controlnet model")
        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE,
            torch_dtype=torch.float16
        )
        print("Loading better VAE")
        better_vae = AutoencoderKL.from_pretrained(
            VAE_CACHE,
            torch_dtype=torch.float16
        )
        print("Loading sdxl")
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
        t2 = time.time()
        print("Setup took: ", t2 - t1)

        upsampler, codeformer_net, device = get_codeformer_models()
        self.upsampler = upsampler
        self.codeformer_net = codeformer_net
        self.device = device

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def resize_to_allowed_dimensions(self, width, height):
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

    def img2img_cany(self, image, prompt, negative_prompt, num_inference_steps, condition_scale, strength, generator):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        images = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=image,
            controlnet_conditioning_scale=condition_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            strength=strength,
        ).images
        return images
        
        

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        src_image: Path = Input(
            description="Input source image for face swap",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="low quality, bad quality, sketches",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        condition_scale: float = Input(
            description="controlnet conditioning scale for generalization",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        strength: float = Input(
            description="Strength of noise in init image",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        seed: int = Input(
            description="Random seed. Set to 0 to randomize the seed", default=0
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed is None) or (seed <= 0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")


        image = self.load_image(image)
        src_image = self.load_image(src_image)
        image_width, image_height = image.size
        print("Original width:"+str(image_width)+", height:"+str(image_height))
        new_width, new_height = self.resize_to_allowed_dimensions(image_width, image_height)
        print("new_width:"+str(new_width)+", new_height:"+str(new_height))
        image = image.resize((new_width, new_height))
        src_image = src_image.resize((new_width, new_height))
        img = image.copy()

        images = self.img2img_cany(
            img, prompt, negative_prompt, num_inference_steps, condition_scale, strength, generator
        )

        args = [[src_image, images[0]]]
        gen_imgs = []
        frame_processors = ["face_swapper"]
        for frame_processor in get_frame_processors_modules(frame_processors):
            frame_processor.pre_check()
            for arg in args:
                gen_arr = frame_processor.process_image(arg)
                gen_img = cv2.cvtColor(gen_arr, cv2.COLOR_BGR2RGB)
                gen_imgs.append(gen_img)

        args_codeformer = [
            (np.array(img), True, False, False, 0, 1.0, self.upsampler, self.codeformer_net, self.device)
            for img in gen_imgs
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(inference_codeformer, args_codeformer))
        results = [Image.fromarray(result) for result in results]

        output_path = f"/tmp/output.png"
        results[0].save(output_path)

        return Path(output_path)
