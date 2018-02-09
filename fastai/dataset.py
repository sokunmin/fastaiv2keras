from enum import IntEnum

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from .conv_learner import *
from .transform import *
import numpy as np


def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]

def parse_csv_labels(fn, skip_header=True):
    skip = 1 if skip_header else 0
    csv_lines = [o.strip().split(',') for o in open(fn)][skip:]
    fnames = [fname for fname, _ in csv_lines]
    csv_labels = {a:b.split(' ') for a,b in csv_lines}
    all_labels = sorted(list(set(p for o in csv_labels.values() for p in o)))
    label2idx = {v:k for k,v in enumerate(all_labels)}
    return sorted(fnames), csv_labels, all_labels, label2idx

def csv_source(folder, csv_file, skip_header=True, suffix='', continuous=False):
    fnames,csv_labels,all_labels,label2idx = parse_csv_labels(csv_file, skip_header)
    full_names = [os.path.join(folder,fn+suffix) for fn in fnames]
    if continuous:
        label_arr = np.array([csv_labels[i] for i in fnames]).astype(np.float32)
    else:
        label_arr = nhot_labels(label2idx, csv_labels, fnames, len(all_labels))
        is_single = np.all(label_arr.sum(axis=1)==1)
        if is_single: label_arr = np.argmax(label_arr, axis=1)
    return full_names, label_arr, all_labels

class ImageClassifierData:
    @classmethod
    def from_paths(self, path, bs=64, tfms=None, trn_name='train', val_name='valid', sz=None, **kwargs):
        '''
        @args:
            path: directory where data resides
            bs: batch size
            tfms: transformations to perform tuple of Keras ImageDataGenerator's (train,validation)
            trn_name: subfolder name of training data (path+trn_name)
            val_name: subfolder name of validation data (path+val_name)
            **kwargs to pass to keras flow_from_directory function
        @returns:
            a tuple of image batch generators (train, validation)
        '''
        if sz == None:
            try:
                sz = tfms[0].sz
            except AttributeError:
                raise Exception('sz must be set through tfms_from_model or passed as an argument')

        if tfms == None: tfms = ImageDataGenerator(), ImageDataGenerator()
        train = tfms[0].flow_from_directory(path + trn_name, target_size=(sz, sz),
                                            class_mode='categorical', shuffle=True, batch_size=bs, **kwargs)
        valid = tfms[0].flow_from_directory(path + val_name, target_size=(sz, sz),
                                            class_mode='categorical', shuffle=False, batch_size=bs, **kwargs)
        return train, valid


class CropType(IntEnum):
    """ Type of image cropping.
    """
    RANDOM = 1
    CENTER = 2
    NO = 3


def tfms_from_model(model, sz, crop_type=CropType.CENTER, **kwargs):
    '''
    @args:
        model: architecture
        target" size image
        crop_type: type of cropping
        **kwargs: other arguments to be passed into ImageDataGenerator
    @returns:
        a tuple of train and validation Keras ImageDataGenerators
    '''
    if model == ResNet50:
        preprocess = preprocess_input
    crop = CenterCrop(sz, preprocess=preprocess)
    if crop_type == CropType.RANDOM:
        crop = RandCrop(sz, preprocess=preprocess)
    elif crop_type == CropType.NO:
        crop = None

    train_gen = ImageDataGenerator(preprocessing_function=crop, **kwargs)
    val_gen = ImageDataGenerator(preprocessing_function=CenterCrop(sz, preprocess=preprocess))
    # train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, **kwargs)
    # val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen.sz = sz
    val_gen.sz = sz
    return train_gen, val_gen
