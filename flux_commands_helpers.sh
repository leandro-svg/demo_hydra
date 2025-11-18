module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load PyTorch/2.6.0-foss-2024a-CUDA-12.6.0
module load TensorFlow/2.18.1-foss-2024a-CUDA-12.6.0
module load LibTIFF/4.6.0-GCCcore-13.3.0

export LD_LIBRARY_PATH=$EBROOTLIBTIFF/lib:$LD_LIBRARY_PATH

# Remove the problematic user-installed Pillow
rm -rf ~/.local/lib/python3.12/site-packages/PIL*
rm -rf ~/.local/lib/python3.12/site-packages/Pillow*

# Create venv WITH --system-site-packages to access module-installed PyTorch & Pillow
python3 -m venv ~/demo/demo_venv_flux/venv --system-site-packages
source ~/demo/demo_venv_flux/venv/bin/activate

# Verify PyTorch is accessible
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} from modules')"

# Verify Pillow works with LibTIFF
python3 -c "from PIL import Image; print('✓ Pillow works with LibTIFF')"

# Verify diffusers works (already installed in your user site-packages)
python3 -c "from diffusers import FluxPipeline; print('✓ FluxPipeline ready')"

# Run flux.py
cd ~/demo/KITTI-Dataset
python3 flux.py