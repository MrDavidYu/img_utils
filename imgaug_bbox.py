import os
from glob import glob

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tqdm

from voc_utils import VOC as voc
import voc_utils as hvu

######################################################################################
# REQUIRES USER INPUT
######################################################################################
ROOT_DIR = "C:\\Users\\davi9801\\Desktop\\HappyUtils\\test_dir\\sample"
EPOCHS = 1  # how many epochs (iterations over whole data)
ITERS = 5  # how many iterations of imgaug per image per epoch
out_path_img = "C:\\Users\\davi9801\\Desktop\\HappyUtils\\test_dir\\sample_out_imgs"
out_path_label = "C:\\Users\\davi9801\\Desktop\\HappyUtils\\test_dir\\sample_out_annotations"

images_path = os.path.join(ROOT_DIR,"JPEGImages")
labels_path = os.path.join(ROOT_DIR,"Annotations")
images = glob(os.path.join(images_path,"*.jpg"))
labels = glob(os.path.join(labels_path,"*.xml"))
######################################################################################

assert len(images) == len(labels)

voc_obj = voc(images_path, labels_path)
voc_obj.init_voc_datastruct()

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 50% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=["constant"],
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=["constant"] # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 0.4), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 0.2), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.0, 0.3)),
                    iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


# Actually write to file
# image_ids = np.random.choice(dataset_train.image_ids, 1)

for epoch in range(EPOCHS):
    repeat_counter = 0  # how many times an image has been repeatedly augmented
    print("epoch: {}".format(repeat_counter))
    for image_path in tqdm.tqdm(images):
        image = np.asarray(Image.open(image_path))
        filename = image_path.split('\\')[-1][:-4]
        bboxes = voc_obj.get_bboxes(filename)
        ia_bboxes = list()
        for bbox in bboxes:
            ia_bbox = ia.BoundingBox(x1=int(bbox[1]), y1=int(bbox[2]), x2=int(bbox[3]), y2=int(bbox[4]), label=bbox[0])
            ia_bboxes.append(ia_bbox)
        bbs = ia.BoundingBoxesOnImage(ia_bboxes, shape=image.shape)

        # Augment images and heatmaps.
        images_aug = []
        bbs_aug = []
        for _ in range(ITERS):
            seq_det = seq.to_deterministic()  # ensure deterministic results (but only for this instance of seq (seq_det))
            images_aug.append(seq_det.augment_image(image))
            bb_aug = seq_det.augment_bounding_boxes([bbs])[0]
            bbs_aug.append(bb_aug.remove_out_of_image().clip_out_of_image())
   
        aug_counter = 0  # iteration no. associated with a repeat_counter
        for image_aug, bb_aug in zip(images_aug, bbs_aug):  # loop over the 5 augmentations
            write_bboxes = list()
            for bbox_obj in bbs.bounding_boxes:
                xmin = bbox_obj.x1
                ymin = bbox_obj.y1
                xmax = bbox_obj.x2
                ymax = bbox_obj.y2
                write_class = bbox_obj.label
                write_bboxes.append([write_class, xmin, ymin, xmax, ymax])

            write_bboxes_aug = list()
            for bbox_obj in bb_aug.bounding_boxes:
                xmin = bbox_obj.x1
                ymin = bbox_obj.y1
                xmax = bbox_obj.x2
                ymax = bbox_obj.y2
                write_class = bbox_obj.label
                write_bboxes_aug.append([write_class, xmin, ymin, xmax, ymax])

            new_name = filename+"_"+str(epoch)+"_"+str(repeat_counter)+"_"+str(aug_counter)
            new_name_img = new_name+"_o.png"  # original image
            new_name_img_aug = new_name+".png"
            new_name_label = new_name+"_o.xml"  # original label
            new_name_label_aug = new_name+".xml"
            if(aug_counter==0):  # for every aug cycle (ITERS augmentations), allow only 1 original output
                cv2.imwrite(os.path.join(out_path_img,new_name_img),image)
                hvu.write_VOC_using_bbox(new_name_img, os.path.join(out_path_label,new_name_label), 1920, 1080, write_bboxes)
            cv2.imwrite(os.path.join(out_path_img,new_name_img_aug),image_aug)
            hvu.write_VOC_using_bbox(new_name_img_aug, os.path.join(out_path_label,new_name_label_aug), 1920, 1080, write_bboxes_aug)
            aug_counter += 1
    repeat_counter += 1
