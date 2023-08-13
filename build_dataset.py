"""Split datasets into train/val/test and resize images if needed.

The IIT Delhi palmprint dataset comes into the following format:
    Left Hand/
        001_1.JPG
        ...
    Right Hand/
        001_1.JPG
        ...
    Segmented/
        Left/
            001_1.bmp
            ...
        Right/
            001_1.bmp
            ...

Original images have size (1600, 1200) and (128,128).

The Tongji contactless palmprint dataset comes into the following format:
    Original Images/
        sessions1/
            00001.tiff
            ...
        sessions2/
            00001.tiff
            ...
    ROI/
        sessions1/
            00001.bmp
            ...
        sessions2/
            00001.bmp
            ...

Original images have size (800, 600) and (128,128).

The REST palmprint dataset comes into the following format:
    p1/
        hand/
            Left/
                p1_l_1.jpg
                ...
                p1_l_5.jpg
            Right/
                p1_r_1.jpg
                ...
                p1_r_5.jpg
    ...
    p179/
        hand/
            Left/
                p179_l_1.jpg
                ...
                p179_l_5.jpg
            Right/
                p179_r_1.jpg
                ...
                p179_r_5.jpg

Original images have size (2048, 1536).
python3 build_dataset.py --data_dir 'data/raw/iit-delhi-palmprint-database-version-10' --dataset "IIT Delhi V1 Segmented" --output_dir "data/IIT Delhi V1"
python3 build_dataset.py --data_dir 'data/raw/iit-delhi-palmprint-database-version-10' --dataset "IIT Delhi V1 Segmented" --output_dir "data/IIT Delhi V1"
python3 build_dataset.py --data_dir 'data/raw/tongji-contactless-palmprint-dataset' --dataset "Tongji Segmented" --output_dir "data/Tongji"
python3 build_dataset.py --data_dir 'data/raw/tongji-contactless-palmprint-dataset' --dataset "Tongji" --output_dir "data/Tongji"
python3 build_dataset.py --data_dir 'data/raw/rest-database' --dataset "REST" --output_dir "data/REST"
"""

import argparse
import random
import os
import glob
import shutil

from PIL import Image
from tqdm import tqdm

HEIGHT = 720
WIDTH = 1280

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/raw', help="Directory with the datasets")
parser.add_argument('--dataset', choices=['IIT Delhi V1','IIT Delhi V1 Segmented','Tongji','Tongji Segmented','REST'], help="Choose between datasets")
parser.add_argument('--output_dir', default='data/', help="Where to write the new data")
parser.add_argument('--resize', default=False, help="Enabling resize")


def resize_and_save(filename, output_dir, new_file_name = '', h=HEIGHT, w=WIDTH):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((h, w), Image.BILINEAR)
    if new_file_name == '':
        image.save(os.path.join(output_dir, filename.split('/')[-1]))
    else:
        image.save(os.path.join(output_dir, new_file_name))
    

class DatasetBuilder:
    def __init__(self, args):
        self.output_dir = args.output_dir
        self.dataset = args.dataset 
        self.data_dir = args.data_dir
        self.filenames = [] # all image links are stored here
        self.resize = args.resize 
        self.new_file_name = ''
    def split(self):
        # Get the filenames
        filenames = self.filenames
        random.seed(230)
        filenames.sort()
        train_filenames = []
        val_filenames = []
        test_filenames = []
        if self.dataset == "Tongji" or self.dataset == "Tongji Segmented":
            for i in range(600):
                choices = list(range(10*i,10*i+10))
                choices.extend(list(range(10*i+6000,10*i+6010)))
                random.shuffle(choices)
                for choice in choices[0:16]:
                    train_filenames.append(filenames[choice])
                for choice in choices[16:18]:
                    val_filenames.append(filenames[choice])
                for choice in choices[18:20]:
                    test_filenames.append(filenames[choice])
        elif self.dataset == "IIT Delhi V1": 
            identities = list(set([filename.split('/')[-2]+"/"+filename.split('/')[-1].split("_")[0] for filename in filenames]))
            for idx in identities:
                choices = glob.glob(f'{self.data_dir}/{idx}*.JPG')
                random.shuffle(choices)
                val_filenames.extend(choices[0:2])
                test_filenames.extend(choices[0:2])
                train_filenames.extend(choices[2:])
        elif self.dataset == "IIT Delhi V1 Segmented":
            identities = list(set([filename.split('/')[-2]+"/"+filename.split('/')[-1].split("_")[0] for filename in filenames]))
            for idx in identities:
                choices = glob.glob(f'{self.data_dir}/Segmented/{idx}*.bmp')
                random.shuffle(choices)
                val_filenames.extend(choices[0:2])
                test_filenames.extend(choices[0:2])
                train_filenames.extend(choices[2:])
        elif self.dataset == "REST":
            identities = list(set([filename.split('/')[-4]+"/"+filename.split('/')[-3]+"/"+filename.split('/')[-2] for filename in filenames]))
            for idx in identities:
                choices = glob.glob(f'{self.data_dir}/REST database/{idx}/*.jpg')
                random.shuffle(choices)
                val_filenames.extend(choices[0:2])
                test_filenames.extend(choices[0:2])
                train_filenames.extend(choices[2:])
        self.filenames = {'train': train_filenames,
                          'val': val_filenames,
                          'test': test_filenames}
    def rename(self, filename):
        pass
    def build(self):
        # Preprocess train, val and test
        for split in ['train', 'val', 'test']:
            output_dir_split = os.path.join(self.output_dir, f'{split}_{self.dataset}')
            assert not os.path.exists(output_dir_split), f'Directory {output_dir_split} is not empty'
            os.mkdir(output_dir_split)
            print(f'Processing {split} data of {self.dataset}, saving preprocessed data to {output_dir_split}')
            for filename in tqdm(self.filenames[split]):
                self.rename(filename)
                if self.resize:
                    resize_and_save(filename, output_dir_split, self.new_file_name, h=HEIGHT, w=WIDTH)
                else:
                    shutil.copyfile(filename, os.path.join(output_dir_split, self.new_file_name))
        print("Done building dataset")

    
class IITDelhiV1Builder(DatasetBuilder):
    def __init__(self, args):
        super().__init__(args)
        if args.dataset == "IIT Delhi V1":
            self.filenames = glob.glob(f'{self.data_dir}/*/*.JPG')
        elif args.dataset == "IIT Delhi V1 Segmented":
            self.filenames = glob.glob(f'{self.data_dir}/Segmented/*/*.bmp')
        assert len(self.filenames)>0, "Couldn't find the dataset at {}".format(self.data_dir)
    def rename(self, filename):
        if self.dataset == "IIT Delhi V1":
            _ = filename.split('/')[-2].split(' ')[0]
            __ = filename.split('/')[-1]
            idx = 1 if _ == "Right" else 2
            idx = str(int(__.split('_')[0])*idx)
            self.new_file_name = f'{idx}_{_}_p{__}'
        elif self.dataset == "IIT Delhi V1 Segmented":
            _ = filename.split('/')[-2]
            __ = filename.split('/')[-1]
            idx = 1 if _ == "Right" else 2
            idx = str(int(__.split('_')[0])*idx)
            self.new_file_name = f'{idx}_{_}_p{__}'
        return super().rename(filename)
        
        
class TongjiBuilder(DatasetBuilder):
    def __init__(self, args):
        super().__init__(args)
        if args.dataset == "Tongji":
            self.filenames = glob.glob(f'{self.data_dir}/Original Images/*/*.tiff')
        elif args.dataset == "Tongji Segmented":
            self.filenames = glob.glob(f'{self.data_dir}/ROI/*/*.bmp')
        assert len(self.filenames)>0, "Couldn't find the dataset at {}".format(self.data_dir)
    def rename(self, filename):
        index = filename.split('/')[-1].split('.')[0]
        _ = int((int(index)-1)/20)+1
        __ = (int(index)-1)%10+1
        ___ = 'Left' if (int(((int(index)-1)/10))%2) else 'Right'
        ____ = filename.split('/')[-2]
        _____ = filename.split('/')[-1]
        idx = 1 if ___ == "Right" else 2
        idx = str(int(_)*idx)
        self.new_file_name = f'{idx}_{___}_p{_}_{____}_{__}_{_____}'
        return super().rename(filename)
        
        
class RESTBuilder(DatasetBuilder):
    def __init__(self, args):
        super().__init__(args)
        self.filenames = glob.glob(f'{self.data_dir}/REST database/*/*/*/*.jpg')
        assert len(self.filenames)>0, "Couldn't find the dataset at {}".format(self.data_dir)
    def rename(self, filename):
        _ = filename.split('/')[-2]
        __ = filename.split('/')[-4]
        ___ = filename.split('/')[-1]
        idx = 1 if _ == "Right" else 2
        idx = str(int(__[1:])*idx)
        self.new_file_name = f'{idx}_{_}_{__}_{___}'
        return super().rename(filename)
        
    
def main():
    args = parser.parse_args()
    
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))
    if args.dataset == 'IIT Delhi V1' or args.dataset == 'IIT Delhi V1 Segmented':
        builder = IITDelhiV1Builder(args)
    if args.dataset == 'Tongji' or args.dataset == 'Tongji Segmented':
        builder = TongjiBuilder(args)
    if args.dataset == 'REST':
        builder = RESTBuilder(args)
    builder.split()
    builder.build()


if __name__ == '__main__':
    main()