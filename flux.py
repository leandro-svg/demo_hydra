import torch
from diffusers import StableDiffusionPipeline

# This model doesn't require authentication
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world, photorealistic, high quality"
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("generated_image.png")
print("✅ Image generated successfully: generated_image.png")