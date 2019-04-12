import os
from os.path import isdir, exists, abspath, join
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class fliphorizontal():
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def forward(self):
        img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        label = self.label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label

class flipvertical():
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def forward(self):
        img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        label = self.label.transpose(Image.FLIP_TOP_BOTTOM)
        return img, label

class rotateimg():
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def forward(self):
        angles = [90, 180, 270]
        angle = np.random.choice(angles, 1)
        img = self.img.rotate(angle)
        label = self.label.rotate(angle)
        return img, label

class gammacorrection():
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def forward(self):
        img = self.img / self.img.max()
        img = img ** 2.2
        img = img * 255.
        return img, self.label

class elasticdeformation():
    def __init__(self, img, label):
        self.img = img
        self.label = label
        self.sigma = 7

    def forward(self):
        alpha_choices = np.arange(50, 125)
        alpha = np.random.choice(alpha_choices)
        shape = self.img.shape
        random_state = np.random.RandomState(None)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        img = map_coordinates(self.img, indices, order=1, mode='reflect').reshape(shape)
        return img, self.label

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1, dataargumentation=False):
        self.batch_size = batch_size
        self.test_percent = test_percent
        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')
        self.files = os.listdir(self.data_dir)
        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]
        self.dataargumentation = dataargumentation

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            current += 1
            data_image = Image.open(self.data_files[current])
            data_image = data_image.resize((388, 388))
            label_image = Image.open(self.label_files[current])
            label_image = label_image.resize((388, 388))
            augseed = np.arange(10)
            augornot = np.random.choice(augseed)
            if self.dataargumentation:
                para = 2
            else:
                para = 20
            if augornot > para:
                data_image, label_image = self.applyDataAugmentation(data_image,label_image)
            else:
                data_image, label_image = self.overlap(data_image, label_image)
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

    def applyDataAugmentation(self,img,label):
        choices1 = [fliphorizontal(img, label), flipvertical(img, label), rotateimg(img, label)]
        num_choice1 = np.arange(1,3)
        num1 = np.random.choice(num_choice1)
        augmentation_choice1 = np.random.choice(choices1, num1)
        for i in augmentation_choice1:
            img, label = i.forward()
        img = np.array(img, dtype=np.float64)
        label = np.array(label, dtype=np.float64)
        choices2 = [gammacorrection(img, label),elasticdeformation(img, label)]
        num_choice2 = np.arange(1,3)
        num2 = np.random.choice(num_choice2)
        augmentation_choice2 = np.random.choice(choices2, num2)
        for i in augmentation_choice2:
            img, label = i.forward()
        img, label = self.overlap(img, label)
        return img, label


    def elasticdeformation(self, img, label, sigma=7):
        alpha_choices = np.arange(50, 125)
        alpha = np.random.choice(alpha_choices)
        shape = img.shape
        random_state = np.random.RandomState(None)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        x, y= np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        img = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
        return img, label

    def overlap(self, img, label):
        img = np.array(img, dtype=np.float64)
        label = np.array(label, dtype=np.float64)
        data_image = np.empty([572, 572])
        data_image[92:480, 92:480] = img
        data_image_up = np.flip(img[0:92, :], 0)
        data_image_down = np.flip(img[296:, ], 0)
        data_image[0:92, 92:480] = data_image_up
        data_image[480:, 92:480] = data_image_down
        data_image[:, 0:92] = np.flip(data_image[:, 92:184], 1)
        data_image[:, 480:572] = np.flip(data_image[:, 388:480], 1)
        data_image = data_image / data_image.max()
        return data_image, label
