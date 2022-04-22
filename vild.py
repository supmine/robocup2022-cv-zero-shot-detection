from easydict import EasyDict

import numpy as np
import torch
import clip

from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import patches

import collections
import json
import numpy as np

import os
import os.path as osp

from PIL import Image
from pprint import pprint
from scipy.special import softmax
import yaml


import tensorflow.compat.v1 as tf
import time
import cv2

FLAGS = {
    'prompt_engineering': True,
    'this_is': True,

    'temperature': 100.0,
    'use_softmax': False,
}
FLAGS = EasyDict(FLAGS)


# Global matplotlib settings
SMALL_SIZE = 16  # 10
MEDIUM_SIZE = 18  # 12
BIGGER_SIZE = 20  # 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Parameters for drawing figure.
display_input_size = (10, 10)
overall_fig_size = (18, 24)

line_thickness = 2
fig_size_w = 35
# fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
mask_color = 'red'
alpha = 0.5


def article(name):
    return 'an' if name[0] in 'aeiou' else 'a'


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace('_', ' ').replace('/', ' or ').lower()
    if rm_dot:
        res = res.rstrip('.')
    return res


single_template = [
    'a photo of {article} {}.'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]

model, preprocess = clip.load("ViT-B/32")


def build_text_embedding(categories):
    if FLAGS.prompt_engineering:
        templates = multiple_templates
    else:
        templates = single_template

    run_on_gpu = torch.cuda.is_available()

    with torch.no_grad():
        all_text_embeddings = []
        print('Building text embeddings...')
        for category in tqdm(categories):
            texts = [
                template.format(processed_name(category['name'], rm_dot=True),
                                article=article(category['name']))
                for template in templates]
            if FLAGS.this_is:
                texts = [
                    'This is ' +
                    text if text.startswith(
                        'a') or text.startswith('the') else text
                    for text in texts
                ]
            texts = clip.tokenize(texts)  # tokenize
            if run_on_gpu:
                texts = texts.cuda()
            text_embeddings = model.encode_text(
                texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        if run_on_gpu:
            all_text_embeddings = all_text_embeddings.cuda()
    return all_text_embeddings.cpu().numpy().T


numbered_categories = [{'name': str(idx), 'id': idx, } for idx in range(50)]
numbered_category_indices = {cat['id']: cat for cat in numbered_categories}


def nms(dets, scores, thresh, max_dets=1000):
    """Non-maximum suppression.
    Args:
      dets: [N, 4]
      scores: [N,]
      thresh: iou threshold. Float
      max_dets: int.
    """
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_dets:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        overlap = intersection / \
            (areas[i] + areas[order[1:]] - intersection + 1e-12)

        inds = np.where(overlap <= thresh)[0]
        order = order[inds + 1]
    return keep


def display_image(path_or_array, size=(10, 10)):
    if isinstance(path_or_array, str):
        image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
    else:
        image = path_or_array

    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


class VILD:
    def __init__(self):
        session = tf.Session(graph=tf.Graph())
        saved_model_dir = './image_version2'

        _ = tf.saved_model.loader.load(session, ['serve'], saved_model_dir)
        self.session = session
        robocup_class = ["apple", "orange", "lime", "banana", "onion", "jelly", "cereal box", "potato chip", "instant noodle", "ketchup", "chocolate",
                         "water bottle", "softdrink", "milk carton", "juice", "coffee cup", "tea", "beer", "table", "chair", "sofa", "fridge", "cabinet"]

        category_name_string = ';'.join(robocup_class)
        category_names = [x.strip() for x in category_name_string.split(';')]
        category_names = ['background'] + category_names
        categories = [{'name': item, 'id': idx+1, }
                      for idx, item in enumerate(category_names)]
        self.category_names = category_names
        robocup_embbed = build_text_embedding(categories)
        self.text_features = robocup_embbed

        max_boxes_to_draw = 25
        nms_threshold = 0.6
        min_rpn_score_thresh = 0.9
        min_box_area = 220
        params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area
        self.params = params

    def detect(self, image_path):
        max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area = self.params

        roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = self.session.run(
            ['RoiBoxes:0', 'RoiScores:0', '2ndStageBoxes:0', '2ndStageScoresUnused:0',
                'BoxOutputs:0', 'MaskOutputs:0', 'VisualFeatOutputs:0', 'ImageInfo:0'],
            feed_dict={'Placeholder:0': [image_path, ]})

        roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
        # no need to clip the boxes, already done
        roi_scores = np.squeeze(roi_scores, axis=0)

        detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
        scores_unused = np.squeeze(scores_unused, axis=0)
        box_outputs = np.squeeze(box_outputs, axis=0)
        detection_masks = np.squeeze(detection_masks, axis=0)
        visual_features = np.squeeze(visual_features, axis=0)

        image_info = np.squeeze(image_info, axis=0)  # obtain image info
        image_scale = np.tile(image_info[2:3, :], (1, 2))
        image_height = int(image_info[0, 0])
        image_width = int(image_info[0, 1])

        rescaled_detection_boxes = detection_boxes / image_scale  # rescale

        nmsed_indices = nms(
            detection_boxes,
            roi_scores,
            thresh=nms_threshold
        )

        # Compute RPN box size.
        box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (
            rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

        # Filter out invalid rois (nmsed rois)
        valid_indices = np.where(
            np.logical_and(
                np.isin(np.arange(len(roi_scores), dtype=np.int), nmsed_indices),
                np.logical_and(
                    np.logical_not(np.all(roi_boxes == 0., axis=-1)),
                    np.logical_and(
                        roi_scores >= min_rpn_score_thresh,
                        box_sizes > min_box_area
                    )
                )
            )
        )[0]
        print('number of valid indices', len(valid_indices))

        detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
        detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
        detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
        detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
        rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]

        start_time = time.time()
        #################################################################
        # Compute text embeddings and detection scores, and rank results
        # text_features = build_text_embedding(categories)

        raw_scores = detection_visual_feat.dot(self.text_features.T)
        if FLAGS.use_softmax:
            scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
        else:
            scores_all = raw_scores

        # Results are ranked by scores
        indices = np.argsort(-np.max(scores_all, axis=1))
        indices_fg = np.array(
            [i for i in indices if np.argmax(scores_all[i]) != 0])
        print("--- %s seconds ---" % (time.time() - start_time))
        #################################################################

        n_boxes = rescaled_detection_boxes.shape[0]

        res_bbox = []
        res_label = []
        for anno_idx in indices[0:int(n_boxes)]:
            rpn_score = detection_roi_scores[anno_idx]
            bbox = rescaled_detection_boxes[anno_idx]
            scores = scores_all[anno_idx]
            if np.argmax(scores) == 0:
                continue

            res_bbox.append(bbox)
            res_label.append(self.category_names[np.argmax(scores)])

        return res_label, res_bbox


test_vild = VILD()


image_path = './examples/five_women_and_umbrellas.jpg'
display_image(image_path, size=display_input_size)


res_label, res_bbox = test_vild.detect(image_path)
