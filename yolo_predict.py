import os
from model import config
import random
import colorsys
import numpy as np
import tensorflow as tf
from model.yolo3_model import yolo


class yolo_predictor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        """
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
            obj_threshold: 目标检测为物体的阈值
            nms_threshold: nms阈值
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        hsv_tuples = [(x / len(self.class_names), 1., 1.)for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)


    def _get_class(self):
        """
        Introduction
        ------------
            读取类别名称
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        Introduction
        ------------
            读取anchors数据
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors



    def eval(self, yolo_outputs, image_shape, max_boxes = 20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图片的大小
            max_boxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1 : 3] * 32



        # 对三个尺度的输出获取每个预测box坐标和box的分数，score计算为置信度x类别概率
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis = 0)                 #boxes是三个预测层的_boxes的list，concat一下变成[nnn,4]的box
        box_scores = tf.concat(box_scores, axis = 0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype = tf.int32)               #最多一副图中只取20个框
        boxes_ = []
        scores_ = []
        classes_ = []


        #对每个类别进行遍历，而不是对位置遍历，这样的好处是同一个框可以预测多个类别的目标，不一定一个网格就一个目标
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])                     #抽出第c类的满足阈值条件的k个框，构成list  k*【4】
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])     #抽出第c类的满足阈值条件的k个类别概率，构成list k*[1]
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = self.nms_threshold)    #传入某个类别的多个框，每个框对应的概率，以及最大输出框的个数，iou阈值就可以使用tf提供的非极大值抑制函数了
            class_boxes = tf.gather(class_boxes, nms_index)                           #使用非极大值抑制得到是框的索引值，配合使用tf.gather方法可以得到非极大值抑制后的框，gather之后shape【nn,4】
            class_box_scores = tf.gather(class_box_scores, nms_index)

            classes = tf.ones_like(class_box_scores, 'int32') * c                    #依据非极大值抑制后的class_box_scores的shape 为[nn]，产生【nn】个类别号
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis = 0)                                        #因为是list，所以concat成张量
        scores_ = tf.concat(scores_, axis = 0)
        classes_ = tf.concat(classes_, axis = 0)
        return boxes_, scores_, classes_                                            #输出非极大值抑制后的目标框，以及对应框的类别、以及概率，注意目标框的shape是【N,4】,每个框形式为【y1,x1,y2,x2】


    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)                                #输入的图像是经过pading resize到416 416的，这里需要把坐标框的值对应到图像原始大小
        boxes = tf.reshape(boxes, [-1, 4])                                           #correct_box输出的boxes的shape为[1,h,w,3,4]转换为【n,4】的框，每一行就是一个框
        box_scores = box_confidence * box_class_probs                                #correct_box输出的box_confidence的shape为[1,h,w,3,1]，box_class_probs的shape为[1,h,w,3,80]利用传播机制
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores                                                     #输出的框对应到原始图像尺寸 boxes【n,4】  box_scores[n,80]


    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标，因为输入图像是经过resize且填充之后输入网络的，最后输出的时候不需要填充的那块图像
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入的大小
            image_shape: 图片的大小
        Returns
        -------
            boxes: 物体框的位置
        """
        box_yx = box_xy[..., ::-1]              #转换一下顺序，方便计算   box_yx的shape为 [None h,w,3,2]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype = tf.float32)
        image_shape = tf.cast(image_shape, dtype = tf.float32)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))     #输入图像按照长边按照416对待，短边等比例缩放，例如输入图像为  832*600的图像，按照这一原则，  new_height=600*416/832=300, new_width =832*416/832=416
        offset = (input_shape - new_shape) / 2. / input_shape                            #实际图像在变换后的图像中左上角的偏移量（相对值）
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale                                               #乘以scale主要考虑到短边的比例映射到【0,1】之间
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)                               #box_mins的shape为 [None h,w,3,2]
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],   #y1     x,y之所以顺序反过来，是因为后面tf里非极大值抑制定义的是这个顺序。。。。
            box_mins[..., 1:2],   #x1
            box_maxes[..., 0:1],  #y2
            box_maxes[..., 1:2]   #x2
        ], axis = -1)
        boxes *= tf.concat([image_shape, image_shape], axis = -1)                     #乘以image_shape对应到原始输入图像上
        return boxes



    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出
            anchors: anchors的位置
            num_classes: 类别数量
            input_shape: 输入大小
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])    #[1 1 1 3 2]
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)

        #网络输出层prediction 在计算损失的时候是怎么处理的，这里要做相同的处理

        # 将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化为占416的比例     实际w = exp(w)*a_w 对应像素值   exp(w)*a_w/input_shape 归一化到【0,1】
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)          #input_shape倒序是因为 shape里面顺序是 h w，预测pridiction输出为wh

       #概率值都要经过sigmoid激活才输出，而yolo_v1直接原始数据输出，不太好
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs                                #返回的都是归一化的值


    def predict(self, inputs, image_shape):
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
            inputs: 处理之后的输入图片
            image_shape: 图像原始大小
        Returns
        -------
            boxes: 物体框坐标
            scores: 物体概率值
            classes: 物体类别
        """
        model = yolo(config.norm_epsilon, config.norm_decay, self.anchors_path, self.classes_path, pre_train = False)
        output = model.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training = False)
        boxes, scores, classes = self.eval(output, image_shape, max_boxes = 20)
        return boxes, scores, classes