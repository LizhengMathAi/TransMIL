# [Step 1] Install all required packages
conda create -n transmil python=3.7 -y
conda activate transmil
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2   -f https://download.pytorch.org/whl/torch_stable.html
wget https://github.com/szc19990412/TransMIL/archive/refs/heads/main.zip
pip install -r requirements.txt
python -m pip install pytorch-toolbelt
python -m pip install torchmetrics


# [Step 2] Create fake dataset for debugging
mkdir -p Camelyon16/pt_files
mkdir -p dataset_csv/camelyon16
# Copy my "make_toy_data.py" file to the repository directory to proceed.
python make_toy_data.py

# [Step 3] Copy my modified "model_interface.py" file to replace the old file in their repository, or you can unzip my .zip file for continued testing.

# -------- Important --------
# [Step 4] If you run the `train.py` directly, you will get an `inplace` Error! Because the package `nystrom_attention` already stop updating, we must fix this issue manually.
# You should be able to find the file `nystrom_attention/nystrom_attention.py` in the conda package dictionary, and then replace the code `q *= self.scale` with `q = q * self.scale`.

# [Step 5] Run the training script
python train.py --stage train --config Camelyon/TransMIL.yaml --gpus 0 --fold 0
