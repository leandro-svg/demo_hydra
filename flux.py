import torch
import time
import psutil
import os
from diffusers import StableDiffusionPipeline

def print_system_info():
    print("=" * 60)
    print("🚀 HPC CLUSTER PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    # System information
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # GPU information
    if torch.cuda.is_available():
        print(f" GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    
    # Cache directory info
    hf_home = os.environ.get('HF_HOME', '~/.cache/huggingface')
    print(f"HuggingFace Cache: {hf_home}")
    print("=" * 60)

def print_gpu_stats(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def print_performance_stats(load_time, inference_time, total_time):
    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"Model Loading Time: {load_time:.2f} seconds")
    print(f"Image Generation Time: {inference_time:.2f} seconds")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    
    # Calculate performance metrics
    steps_per_second = 50 / inference_time  # 50 inference steps
    print(f"Inference Speed: {steps_per_second:.2f} steps/second")
    
    # Memory efficiency
    if torch.cuda.is_available():
        gpu_utilization = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100
        print(f"GPU Utilization: {gpu_utilization:.1f}%")
    
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_percent = (psutil.virtual_memory().used / psutil.virtual_memory().total) * 100
    print(f"CPU Usage: {cpu_percent:.1f}%")
    print(f"RAM Usage: {ram_percent:.1f}%")
    print("=" * 60)

# Print initial system information
print_system_info()

# Start timing
start_time = time.time()

print("\n🔄 Loading Stable Diffusion model...")
load_start = time.time()

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

load_time = time.time() - load_start
print(f"Model loaded in {load_time:.2f} seconds!")
print_gpu_stats("After model loading")

prompt = "A cat holding a sign that says hello world, photorealistic, high quality"
print(f"\nGenerating image with prompt: '{prompt}'")
print("Running inference...")

inference_start = time.time()
image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
inference_time = time.time() - inference_start

total_time = time.time() - start_time

image.save("generated_image.png")
print("Image generated successfully: generated_image.png")
print_gpu_stats("After inference")

# Print performance statistics
print_performance_stats(load_time, inference_time, total_time)
