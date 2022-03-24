import os.path as osp
from utils.augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, NormalizeTensor
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt

def make_datapath_list(rootpath):
    original_img_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annotation_img_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    # train, validation
    train_ids = osp.join(rootpath, 'ImageSets/Segmentation/train.txt')
    val_ids = osp.join(rootpath, 'ImageSets/Segmentation/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_ids):
        img_id = line.strip()
        img_path = (original_img_template % img_id)
        anno_path = (annotation_img_template % img_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_ids):
        img_id = line.strip()
        img_path = (original_img_template % img_id)
        anno_path = (annotation_img_template % img_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train':Compose([
                Scale(scale=[0.5, 1]),
                RandomRotation(angle=[-10,10]),
                RandomMirror(),
                Resize(input_size),
                NormalizeTensor(color_mean, color_std)
            ]),

            'val': Compose([
                Resize(input_size),
                NormalizeTensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        img_path_file = self.img_list[index]
        anno_path_file = self.anno_list[index]
        
        img = Image.open(img_path_file)
        anno_class_img = Image.open(anno_path_file)

        img, anno_class_img = self.transform(self.phase, img, anno_class_img )

        return img, anno_class_img



if __name__ == '__main__':
    rootpath = 'D:/Workspace/deep_learning/object_detection_SSD/data/VOCdevkit/VOC2012/'
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
    # print(len(val_img_list))

    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = MyDataset(img_list=train_img_list, 
                              anno_list=train_anno_list, 
                              phase='train', 
                              transform= DataTransform(input_size=475,color_mean=color_mean, color_std=color_std))

    
    val_dataset = MyDataset(img_list=val_img_list, 
                            anno_list=val_anno_list, 
                            phase='val', 
                            transform= DataTransform(input_size=475,color_mean=color_mean, color_std=color_std))

    # print(f'img_shape:{val_dataset[0][0].shape}')
    # print(f'anno_shape:{val_dataset[0][1].shape}')

    batch_size = 4
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False)

    dataloader_dict = {
        'train': train_loader,
        'val': val_loader
    }

    imgs, labels = next(iter(dataloader_dict['train']))
    print(imgs.shape)

    # img = imgs[0].numpy().transpose(1,2,0)
    # plt.imshow(img.astype('uint8'))
    # plt.show()

    # label = labels[0].numpy()
    # plt.imshow(label)
    # plt.show()