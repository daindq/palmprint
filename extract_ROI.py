import glob
from tqdm import tqdm
from ROI_extract.PROIE import PROIE
import os
filenames = glob.glob('data/raw/rest-database/REST database/*/Hand/*/*.jpg')
proie = PROIE()
save_dir = "data/raw/rest-database/REST ROI"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for file in tqdm(filenames):
    proie.extract_roi(file, rotate=False)
    proie.save(f'{save_dir}/{file.split("/")[-1]}')
filenames = glob.glob('data/raw/CASIA-PalmprintV1/CASIA-PalmprintV1/*/*.jpg')
proie = PROIE()
save_dir = "data/raw/CASIA-PalmprintV1/CASIA-PalmprintV1_ROI"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for file in tqdm(filenames):
    proie.extract_roi(file, rotate=False)
    proie.save(f'{save_dir}/{file.split("/")[-1]}')
filenames = glob.glob('data/raw/CASIA-Multi-Spectral-PalmprintV1/CASIA-Multi-Spectral-PalmprintV1/images/*.jpg')
proie = PROIE()
save_dir = "data/raw/CASIA-Multi-Spectral-PalmprintV1/CASIA-Multi-Spectral-PalmprintV1_ROI"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for file in tqdm(filenames):
    proie.extract_roi(file, rotate=False)
    proie.save(f'{save_dir}/{file.split("/")[-1]}')