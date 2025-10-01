# MediQ-GAN: Quantum-Inspired GAN for High Resolution Medical Image Generation

This repository provides code to preprocess datasets, train models, and implement the quantum-inspired GAN in our paper.

> **Note:** Parts of this codebase were cleaned and commented with the assistance of ChatGPT/LLM to improve readability.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/QingyueJ-nd/MediQ-GAN.git
pip install -r requirements.txt
```

## Datasets

Download the datasets using the following links or from the original websites shown in the Data Availability section. 
Please review and follow each dataset's license/terms.

### ISIC2019

```bash
wget https://notredame.box.com/shared/static/uw8g5urs7m4n4ztxfo100kkga6arzi9k.tar -O ISIC_2019_train.tar
tar -xvf ISIC_2019_train.tar
```

### ODIR-5k

```bash
wget https://notredame.box.com/shared/static/4w12lzj9muar74q12s20wfip9c9hn616.gz -O ODIR-5k_Train.tar.gz
tar -xvf ODIR-5k_Train.tar.gz
```

### RetinaMNIST

```bash
pip install medmnist
```

## Preprocessing

Use `preprocess.py` to prepare datasets. The following commands replicate the paper's setup.

### ISIC2019

```bash
python preprocess.py ISIC2019 \
  --root_dir /path/to/ISIC_2019_train \
  --image_dir ISIC_2019_Training_Input \
  --csv_path ISIC_2019_Training_Metadata.csv \
  --output_dir labeled_input \
  --filter_downsampled
```

### ODIR-5k

```bash
python preprocess.py ODIR-5k \
  --root_dir /path/to/ODIR-5k_Train \
  --image_dir preprocessed_images \
  --csv_path full_df.csv \
  --output_dir labeled_input \
  --organize_into_label_dirs
```

### RetinaMNIST

```bash
python preprocess.py RetinaMNIST \
  --out_dir /path/to/output_root \
  --rgb \
  --merge_splits \
  --flatten_train
```

After preprocessing, point `--data_root` to the resulting class-organized folders.

## Training

Main training script: `mediq-gan.py`. For the full list of options:

```bash
python mediq-gan.py --help
```

### Example: ISIC2019 (64×64)

```bash
python hybridgan_simple.py \
  --mode train --version 0 \
  --dataset ISIC2019 \
  --noise_size 128  \
  --encoder_type cnn --decoder_type vanilla \
  --n_qubits 16 --q_depth 8 --n_generators 5 --force_cpu_quantum 0 \
  --feature_split_ratio 0.5 \
  --gan_type wgan-gp --lambda_gp 10.0 \
  --batch_size 10 \
  --use_proto 1  --proto_path prototype/isic2019_avg.pt \
  --lrG_encoder 1e-4 --lrG_decoder 1e-4 --lrD 2e-4 \
  --seed 42
```

### Example: ODIR-5k (Pure classical)

```bash
python hybridgan_simple.py \
  --mode train --version 0 \
  --dataset ODIR-5k \
  --noise_size 128  \
  --encoder_type cnn --decoder_type vanilla \
  --n_generators 0  \
  --feature_split_ratio 0.5 \
  --gan_type wgan-gp --lambda_gp 10.0 \
  --batch_size 10 \
  --use_proto 1  --proto_path prototype/ODIR-5k_avg.pt \
  --seed 42
```

### Example: RetinaMNIST

```bash
python hybridgan_simple.py \
  --mode train --version 0 \
  --dataset RetinaMNIST \
  --noise_size 128  \
  --encoder_type cnn --decoder_type style2 \
  --n_generators 3  \
  --feature_split_ratio 0.25 \
  --gan_type wgan-gp --lambda_gp 10.0 \
  --batch_size 10 \
  --use_proto 1  --proto_path prototype/RetinaMNIST_avg.pt \
  --seed 42
```

## Generating Images

```bash
python hybridgan_simple.py \
  --mode generate --version 0 \
  --dataset RetinaMNIST \
  --noise_size 128  \
  --encoder_type cnn --decoder_type style2 \
  --n_generators 3  \
  --feature_split_ratio 0.25 \
  --gan_type wgan-gp --lambda_gp 10.0 \
  --batch_size 10 \
  --use_proto 1  --proto_path prototype/RetinaMNIST_avg.pt \
  --seed 42
```

## Evaluation (FID / LPIPS)

If evaluation scripts are included in `metrics/`:

### Fréchet Inception Distance (FID):

```bash
python eval_fid.py \
  --real_dir /path/to/real/images \
  --fake_dir outputs/isic_run1/samples \
  --output results/fid_results.txt \
  --dataset_res 64
```

### LPIPS (Intra-class Diversity)

Using standard LPIPS backbone:
```bash
python eval_lpips.py \
  --real_dir /path/to/real/images \
  --gen_dir outputs/isic_run1/samples \
  --output_json results/lpips_results.json \
  --lpips_backbone alex \
  --image_size 64
```
Using custom ResNet18 model:
```bash
python eval_lpips.py \
  --real_dir /path/to/real/images \
  --gen_dir outputs/isic_run1/samples \
  --output_json results/lpips_results.json \
  --custom_model_path lpips_resnet18_ISIC_final.pth \
  --image_size 64
```


## Acknowledgements

* ISIC2019, ODIR-5k, and RetinaMNIST datasets.
* ChatGPT/LLM assistance was used to clean, reorganize, and comment parts of the code for better readability.
