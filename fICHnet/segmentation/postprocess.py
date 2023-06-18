'''
1. Combine the segmentation results
2. Create separate directories for registered and unregistered images
3. Save as compressed file
'''
import os
import argparse
import numpy as np
import nibabel as nib


def empty_index(vec):
    indices = np.where(vec > 1)[0]
    return [min(indices), max(indices)]


def apply_square_bbox(img, square=True):
    if len(img.shape) == 4:
        img = img.sum(3)
    empty_x = empty_index(img.sum(2).sum(1))
    empty_y = empty_index(img.sum(2).sum(0))
    empty_z = empty_index(img.sum(1).sum(0))
    start_xy = min(empty_x[0], empty_y[0])
    end_xy = max(empty_x[1], empty_y[1])
    if square:
        return img[start_xy:end_xy, start_xy:end_xy, empty_z[0]:empty_z[1]]
    else:
        return img[empty_x[0]:empty_x[1], empty_y[0]:empty_y[1]]


def remove_blank(img, HU_thres=1, thres=0.1, return_index=False):
    '''
    Args:
        `HU_thres`: Hounsfield unit above which a pixel is counted as being
        inside brain tissues
        `thres`: percentage of non-zero pixel below which a particular slice is
        removed
    '''
    red_img = apply_square_bbox(np.round(img), square=False)
    H, W = red_img.shape[:2]
    z_perc = (red_img > HU_thres).sum(axis=(0, 1))/(H*W)
    z_perc = (z_perc > thres).astype(float)
    # reduce the threshold if the image is too small
    # if z_perc.max() == 0:
    #     z_perc = (img > HU_thres).sum(axis=(0, 1))/(H*W)
    #     z_perc = (z_perc > thres/10).astype(float)

    N = len(z_perc)//2
    neck = np.where(z_perc[:N] == 0)[0]
    start = max(neck) if len(neck) > 0 else 0
    vertex = np.where(z_perc[N:] == 0)[0]
    end = min(vertex) if len(vertex) > 0 else len(z_perc[N:])

    if return_index:
        return[start+1, end+N]
    else:
        return img[..., (start+1):(end+N)]


def remove_z(img_path, out_path, depth_thres=15, windows=[0, 100]):
    img_ob = nib.load(img_path)
    img = img_ob.get_fdata().clip(windows[0], windows[1])
    img = remove_blank(img)
    with open(os.path.dirname(img_path)+'/log.txt', 'w') as f:
        if img.shape[2] < depth_thres:
            f.writelines('Stop')
        else:
            f.writelines('Continue')
    save_ob = nib.Nifti1Image(img, img_ob.affine, header=img_ob.header)
    nib.save(save_ob, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()
    remove_z(args.img_path, args.out_path)
