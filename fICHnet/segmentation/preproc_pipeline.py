import os
import numpy as np
import skimage
import nibabel as nib
from scipy import ndimage
from fsl.data.image import Image
from fsl.utils.image.resample import resample
from . import postprocess as PP


def normalize(img, window=[0, 100]):
    img = img.clip(window[0], window[1])
    if len(img.shape) == 3:
        return img/window[1]
    elif len(img.shape) == 4:
        img -= img.min(axis=-1, keepdims=True)
        return img/img.max(axis=-1, keepdims=True)


def padding_to_shape(x, to_shape):
    shape_diff = np.array(to_shape) - np.array(x.shape)[:2]
    pad_x = (shape_diff[0]//2, shape_diff[0]//2 + shape_diff[0] % 2)
    pad_y = (shape_diff[1]//2, shape_diff[1]//2 + shape_diff[1] % 2)
    return np.pad(x, (pad_x, pad_y, (0, 0)))


def depth_more(img, depths):
    D = img.shape[2]
    start_depth = (D - depths)//2
    return img[..., start_depth:(start_depth + depths)]


def depth_less(img, depths, random=True):
    H, W, D = img.shape
    bottom_num = (depths - D)//2
    top_num = depths - D - bottom_num
    img_list = [np.zeros([H, W, top_num]), img,
                np.zeros([H, W, bottom_num])]
    return np.concatenate(img_list, axis=2)


def select_depth(img, depths):
    if img.shape[2] > depths:
        return depth_more(img, depths)
    elif img.shape[2] == depths:
        return img
    else:
        return depth_less(img, depths)


def LCC(img, window=[0, 100]):
    ori_img = img.copy()
    lab_list = []
    for i in range(img.shape[2]):
        # thresholding and erosion
        mask = (img[:, :, i] > window[0]).astype(float) + \
                (img[:, :, i] < window[1]).astype(float)
        mask = skimage.morphology.binary_erosion(mask == 2)

        # largest connected component
        labels = skimage.measure.label(mask)
        lab_count = np.bincount(labels.flat)
        if len(lab_count) == 1:
            LCC = np.zeros(labels.shape)
        else:
            LCC = (labels == np.argmax(lab_count[1:]) + 1)

        # dilation and fill holes
        LCC = skimage.morphology.binary_dilation(LCC)
        lab_list.append(ndimage.binary_fill_holes(LCC))

    masks = (np.stack(lab_list, axis=-1)).astype(int)
    return np.round(ori_img*masks)


def np2fsl(img, affine):
    rand_ID = "".join([str(i) for i in list(np.random.choice(9, 10))])
    tmp_file = os.getcwd()+"/img{}.nii.gz".format(rand_ID)
    nib_img = nib.Nifti1Image(img, affine)
    nib_img.to_filename(tmp_file)
    new_img = Image(tmp_file)
    return new_img, tmp_file


def preprocess(img_path, target_res=5, sel_depth=18, to_shape=[128, 128]):
    '''
    Method:
    1. resample z axis to 5mm resolution
    2. skullstrip via histogram analysis
    3. remove blank slices
    4. select 18 central slices
    5. remove background pixels
    6. normalize to between 0 and 1
    7. resample the long edge of the image to 128 voxels
    8. zero pad the short edge to 128 voxels
    '''
    img = Image(img_path)
    res = img.pixdim[2]

    # sometimes the pixdim is inaccurate
    # it is not possible for head CT to be smaller than 50 slices yet having
    # 1mm resolution
    abort = img.shape[2] < 50 and res < 1
    if not abort:
        new_shape = [*img.shape[:2], int(img.shape[2]*res/target_res)]
        new_img, affine = resample(img, new_shape)

        ss_img = LCC(new_img)
        noback = PP.remove_blank(ss_img)
        noback = select_depth(noback, sel_depth)
        noback = PP.apply_square_bbox(np.round(noback), square=False)
        noback = normalize(noback)

        noback, tmp_file = np2fsl(noback, affine)
        max_size = max(noback.shape[0:2])
        new_shape = [*to_shape, noback.shape[2]]
        new_shape[0] = int(noback.shape[0]/max_size*to_shape[0])
        new_shape[1] = int(noback.shape[1]/max_size*to_shape[1])
        new_img, affine = resample(noback, new_shape)
        new_img = padding_to_shape(new_img, to_shape)
        os.remove(tmp_file)

        # new_nib = nib.Nifti1Image(new_img, affine=affine)
        # new_nib.to_filename(os.getcwd() + "/" + os.path.basename(img_path))
        return new_img
    else:
        return None
