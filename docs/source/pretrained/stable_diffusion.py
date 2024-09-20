import torch
from diffusers import StableDiffusion3Pipeline

CACHE_DIR = 'pretrained/stable_diffusion/'
PRETRAINED = (
    CACHE_DIR
    + 'stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp8.safetensors'
)
PROMPT = 'a picture of a cat holding a sign that says hello world'

pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_single_file(
    PRETRAINED,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
)
pipe.to('cuda')
output = pipe([PROMPT, PROMPT])
image, = output.images
image.save('sd3-single-file-t5-fp8.png')
