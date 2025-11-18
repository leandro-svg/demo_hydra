set -euxo pipefail


module purge
module load Python/3.12.3-GCCcore-13.3.0

# If needed to switch python version
# module spider python
# module load Python/3.13.1-GCCcore-14.2.0
# module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0
module load PyTorch/2.6.0-foss-2024a-CUDA-12.6.0
module load TensorFlow/2.18.1-foss-2024a-CUDA-12.6.0


python3 -m venv ~/demo/demo_venv_etro/venv
source ~/demo/demo_venv_etro/venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir numpy plotly ipykernel tqdm jupyter matplotlib gitpython wandb opencv-python moviepy==1.0.3


# Fix pillow / libtiff issue
module load LibTIFF/4.6.0-GCCcore-13.3.0
export LD_LIBRARY_PATH=$EBROOTLIBTIFF/lib:$LD_LIBRARY_PATH
pip uninstall pillow -y
pip install --no-cache-dir --no-binary=:all: --force-reinstall pillow

# Register the new kernel for Jupyter
python3 -m ipykernel install --user --name demo_venv_18 --display-name "Python (demo_venv_18)"

# Interpretter path to be used in Jupyter notebooks
~/demo/demo_venv_etro/venv/bin/python

# If needed : 
/user/brussel/109/vsc10985/demo/demo_venv_etro/venv/bin/python -m pip install ipykernel -U --force-reinstall

# Download KITTI data
mkdir -p data
cd data

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0048_sync.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0048/2011_09_26_drive_0048_tracklets.zip

unzip data/2011_09_26_drive_0048_sync.zip
unzip data/2011_09_26_calib.zip
unzip data/2011_09_26_drive_0048_tracklets.zip


# Diffusion part
pip install transformers
pip install diffusers 
