import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        #############################################
        # TODO Initialize  Dataset
        #############################################
        self.image_paths = image_paths
        self.transform = transform


    def __getitem__(self, idx):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################
        # image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        print ("earliest: ", img.shape)
        if self.transform:
            img = self.transform({"image": img})["image"]
        
        transed_img = torch.from_numpy(img)
        return transed_img
    

    def __len__(self):
        return len(self.image_paths)


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        # images_b = list(zip(*batch))
        print (len(batch))
        out_batch = dict()
        out_batch['images'] = torch.stack(batch)
        print ("here: ", out_batch['images'].shape)
        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


        

 