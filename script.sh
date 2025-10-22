set -euxo pipefail


module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load TensorFlow/2.13.0-foss-2023a


python3 -m venv ~/demo/demo_venv_hydra/venv --system-site-packages
source ~/demo/demo_venv_hydra/venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir ipykernel tqdm jupyter gitpython wandb pykitti opencv-python moviepy==1.0.3
python -m ipykernel install --user --name demo_venv_hydra --display-name "Python (demo_venv_hydra)"




wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0048_sync.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0048/2011_09_26_drive_0048_tracklets.zip

unzip data/2011_09_26_drive_0048_sync.zip
unzip data/2011_09_26_calib.zip
unzip data/2011_09_26_drive_0048_tracklets.zip
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDAARCH=80


cd ~/TrajFlow
python setup/setup_trajflow.py develop