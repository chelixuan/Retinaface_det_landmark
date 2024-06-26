import cv2
import math
import numpy as np
from itertools import product as product

from rknn.api import RKNN


IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640
MEAN = (104, 117, 123)
MIN_SIZES = [[16, 32], [64, 128], [256, 512]]
STEPS = [8, 16, 32]
VARIANCE = [0.1, 0.2]
NMS_CONFIDENCE_THRESHOLD = 0.02
NMS_THRESHOLD = 0.4
VIS_SCORE_THRESHOLD = 0.5

RKNN_MODEL = '/home/chelx/Retinaface_rknn/ckpt/mobilenet0.25_Final_sim_opt-0.rknn'
IMG_PATH = './test.jpg'
RES_PATH = './res_rknn_test.jpg'

def decode(loc, priors, variances):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                             ), axis=1)

    return landms


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = MIN_SIZES
        self.steps = STEPS
        self.image_size = image_size
        self.feature_maps = [[math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)

        return output


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True, verbose_file='./clx_crypt.log')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('load rknn model failed')
        exit(ret)

    # init runtime environment
    ret = rknn.init_runtime(target = 'rk3566', device_id='0f51bb82820a2745')
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    # Set inputs
    ori_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    orig_h, orig_w, _ = ori_img.shape
    img = cv2.resize(ori_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # img = np.float32(img)
    # img -= MEAN
    # img = np.transpose(np.expand_dims(img, 0), (0, 3, 1, 2))
    img = np.expand_dims(img, 0)
    loc, conf, landms = rknn.inference(inputs=[img])

    print()
    print('loc = ', loc)
    print(loc.shape)
    print()
    print('conf = ', conf)
    print(conf.shape)
    print()
    print('landms = ', landms)
    print(landms.shape)
    print()


    priorbox = PriorBox(image_size=(orig_h, orig_w))
    prior_data = priorbox.forward()

    boxes = decode(np.squeeze(loc), prior_data, VARIANCE)
    scale = np.array([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT])
    resize = IMAGE_HEIGHT / orig_h
    boxes = boxes * scale / resize

    scores = np.squeeze(conf)[:, 1]
    landms = decode_landm(np.squeeze(landms), prior_data, VARIANCE)
    scale1 = np.array([IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT,
                       IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT,])
    landms = landms * scale1 / resize

    # ignore low scores
    inds = np.where(scores > NMS_CONFIDENCE_THRESHOLD)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, NMS_THRESHOLD)

    dets = dets[keep, :]
    landms = landms[keep]
    dets = np.concatenate((dets, landms), axis=1)

    print(f'results = {dets} \n')

    # vis_results
    for b in dets:
        if b[4] < VIS_SCORE_THRESHOLD:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        
        print()
        print('b = ', b)
        print()

        cv2.rectangle(ori_img, (b[0], b[1]), (b[2], b[3]), (0, 180, 150), 1)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(ori_img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.circle(ori_img, (b[5], b[6]), 2, (0, 0, 255), -1)
        cv2.circle(ori_img, (b[7], b[8]), 2, (255, 0, 0), -1)
        cv2.circle(ori_img, (b[9], b[10]), 2, (255, 0, 255), -1)
        cv2.circle(ori_img, (b[11], b[12]), 2, (0, 255, 0), -1)
        
    # save image
    cv2.imwrite(RES_PATH, ori_img)

    rknn.release()


