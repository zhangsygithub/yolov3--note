# -*- coding:utf-8 -*-
# author: gaochen
# date: 2018.06.04


import numpy as np
import tensorflow as tf
import os

class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        """
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
            anchors_path: yolo anchor 文件路径
            classes_path: 数据集类别对应文件
            pre_train: 是否使用预训练darknet53模型
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()


    def _get_class(self):
        """
        Introduction
        ------------
            获取类别名字
        Returns
        -------
            class_names: coco数据集类别对应的名字
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
            获取anchors
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)



    def _batch_normalization_layer(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        '''
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
            input_layer: 输入的四维tensor
            name: batchnorm层的名字
            trainging: 是否为训练过程
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            bn_layer: batch normalization处理之后的feature map
        '''
        bn_layer = tf.layers.batch_normalization(inputs = input_layer,
            momentum = norm_decay, epsilon = norm_epsilon, center = True,
            scale = True, training = training, name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)


    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):
        """
        Introduction
        ------------
            使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
            因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            strides: 卷积步长
            name: 卷积层名字
            trainging: 是否为训练过程
            use_bias: 是否使用偏置项
            kernel_size: 卷积核大小
        Returns
        -------
            conv: 卷积之后的feature map
        """
        if strides > 1:
            # 在输入feature map的长宽维度进行padding
            inputs = tf.pad(inputs, paddings = [[0, 0], [1, 0], [1, 0], [0, 0]], mode = 'CONSTANT')
        conv = tf.layers.conv2d(
            inputs = inputs, filters = filters_num,
            kernel_size = kernel_size, strides = [strides, strides], kernel_initializer = tf.glorot_uniform_initializer(),
            padding = ('SAME' if strides == 1 else 'VALID'), kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4), use_bias = use_bias, name = name)
        return conv


    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            trainging: 是否为训练过程
            blocks_num: block的数量
            conv_index: 为了方便加载预训练权重，统一命名序号
            weights_dict: 加载预训练模型的权重
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            inputs: 经过残差网络处理后的结果
        """

        #残差块的输入x
        layer = self._conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = "conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        for _ in range(blocks_num):
            #用shortcut临时记录x
            shortcut = layer
            #1*1的卷积核减少通道维数
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1

            #3*3的卷积核实现常规卷积
            layer = self._conv2d_layer(layer, filters_num, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1

            #残差结果记录为layer也就是f(x)-x  与shortcut相加  得到残差结构
            layer += shortcut
        return layer, conv_index


    def _darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs: 模型输入变量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            weights_dict: 预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
            route1: 返回第26层卷积计算结果52x52x256, 供后续使用
            route2: 返回第43层卷积计算结果26x26x512, 供后续使用
            conv_index: 卷积层计数，方便在加载预训练模型时使用

        Note
        -------
        残差块的卷积层计数：第一层   残差块x1    残差块x2   残差块x8  残差块x8  残差块x4
                             1         1+1*2     1+2*2      1+8*2      1+8*2     1+4*2
                             1         4          9         26         43        52
        """
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1           #每搭建一层网络之后，index索引才会加1

            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index,  filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        return  route1, route2, conv, conv_index


    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率,一个yolo_block氛围7个卷积，从第五个卷积层抽出路由层
        Parameters
        ----------
            inputs: 输入特征
            filters_num: 卷积核数量
            out_filters: 最后输出层的卷积核数量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            route: 返回最后一层卷积的前一层结果
            conv: 返回最后一层卷积的结果
            conv_index: conv层计数
        """
        conv = self._conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        route = conv
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        conv = self._conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index


    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs: 模型的输入变量
            num_anchors: 每个grid cell负责检测的anchor数量
            num_classes: 类别数量
            training: 是否为训练模式
        """
        conv_index = 1
        #搭建darknet53作为特征提取的主干网络   实际应该是conv26,conv43 conv52
        conv2d_26, conv2d_45, conv, conv_index = self._darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        #搭建yolov3的特征融合部分
        with tf.variable_scope('yolo'):       #52+7
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_45], axis = -1, name = 'route_0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training,norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = 'route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]



    def yolo_head(self, feats, anchors, num_classes, input_shape, training = True):
        """
        Introduction
        ------------
            根据不同大小的feature map做多尺度的检测，三种feature map大小分别为13x13x255, 26x26x255, 52x52x255  其中255= num_anchors * (num_classes + 5)
        Parameters
        ----------
            feats: 输入的特征feature map
            anchors: 针对不同大小的feature map的anchor,在外面由mask选择，例如mask=6,7,8，则，anchors=[[116,90],  [156,198],  [373,326]]
            num_classes: 类别的数量
            input_shape: 图像的输入大小，一般为416
            trainging: 是否训练，用来控制返回不同的值
        Returns
        -------
        """
        num_anchors = len(anchors)                                                                          #每个预测层上3种anchor, anchors.shape=[3,2]
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype = tf.float32), [1, 1, 1, num_anchors, 2])    #使用reshape强2D扩充为5D， [1,1,1,3,2]

        # y_true的格式[None,h,w,anchors,labels]  其中anchors个数=3， h=26,w=26,labels=[cx,cy,w,h,obj,p1,p2,...,p80]  其中[cx,cy,w,h]都是相对于输入416*416后归一化到[0,1]的数据
        #feats是某一个检测层的结果，[None,h,w,c]=[None 26,26,255]   其中255= num_anchors * (num_classes + 5)=【[cx,cy,w,h,obj,p1,p2,...,p80]*3】
        grid_size = tf.shape(feats)[1:3]                                                                      #grid_size=[h,w]=[26,26]

        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])       #将输出预测张量转换为[None 26,26,3,85]，prediction预测的是偏移量[0,1]之间

        # 这里构建26*26*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])          #
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)                                                          #axis=-1表示在最低维度上concat [26 26 1 2]
        grid = tf.cast(grid, tf.float32)                                                                       #grid是用range生成的，类型默认为int,需要cast到float32
        #将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)             #predictions[..., :2] 张量为[None 26 26 3 2]   grid张量为[26,26,1,2]使用python的传播机制，先将grid扩展为[26,26,3,2]的形式再相加
                                                                                                              #这里除以grid_size归一化到[0,1],grid_size到序是因为grid_size内容为【h,w】，而prediction的前两位为[x,y]
        #将w,h也归一化为占416的比例
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]                           #同理将w,h先用exp非线性化，在乘以anchor的尺寸进行调节，最后除以输入尺寸归一化
        box_confidence = tf.sigmoid(predictions[..., 4:5])                                                    #obj表征是否为目标的概率，使用sigmoid转换到[0,1]之间
        box_class_probs = tf.sigmoid(predictions[..., 5:])                                                    #同理后面的类别概率做同样的操作
        if training == True:
            return grid, predictions, box_xy, box_wh                                                          #grid不同位置的偏移量，prediction预测的是相对于grid的偏移量，box_xy,box_wh对应于全图，并且归一化到了[0,1]
        return box_xy, box_wh, box_confidence, box_class_probs


    def yolo_boxes_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        """
        Introduction
        ------------
            该函数是将box的坐标修正，除去之前按照长宽比缩放填充的部分，最后将box的坐标还原成相对原始图片的
        Parameters
        ----------
            feats: 模型输出feature map
            anchors: 模型anchors
            num_classes: 数据集类别数
            input_shape: 训练输入图片大小
            image_shape: 原始图片的大小
        """
        input_shape = tf.cast(input_shape, tf.float32)
        image_shape = tf.cast(image_shape, tf.float32)
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape, training = False)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.
        box_max = box_yx + box_hw / 2.
        boxes = tf.concat(
            [box_min[..., 0:1],
             box_min[..., 1:2],
             box_max[..., 0:1],
             box_max[..., 1:2]],
            axis = -1
        )
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        boxes = tf.reshape(boxes, [-1, 4])
        boxes_scores = box_confidence * box_class_probs
        boxes_scores = tf.reshape(boxes_scores, [-1, num_classes])
        return boxes, boxes_scores


    def box_iou(self, box1, box2):
        """
        Introduction
        ------------
            计算box tensor之间的iou
        Parameters
        ----------
            box1: shape=[grid_size, grid_size, anchors, xywh]
            box2: shape=[box_num, xywh]
        Returns
        -------
            iou:
        """
        box1 = tf.expand_dims(box1, -2)     #box1: shape=[grid_size, grid_size, anchors, xywh]，扩展之后变为[grid_size, grid_size, anchors, 1，xywh]
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.
        box1_maxs = box1_xy + box1_wh / 2.

        box2 = tf.expand_dims(box2, 0)   #box2: shape=[box_num, xywh] 扩展之后变为  [1，box_num, xywh]
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.
        box2_maxs = box2_xy + box2_wh / 2.

        intersect_mins = tf.maximum(box1_mins, box2_mins)                        #扩展之后变为[grid_size, grid_size, anchors, num_box，xy]
        intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        iou = intersect_area / (box1_area + box2_area - intersect_area)         #输出[grid_size, grid_size, anchors, num_box]的IOU
        return iou



    def yolo_loss(self, yolo_output, y_true, anchors, num_classes, ignore_thresh = .5):
        """
        Introduction
        ------------
            yolo模型的损失函数
        Parameters
        ----------
            yolo_output: yolo模型的输出  是一个list [conv2d_59, conv2d_67, conv2d_75] 三个张量的shape分别为[None 52,52,255]，[None 26,26,255], [None 13,13,255],
            y_true: 经过预处理的真实标签，shape为[batch, grid_size, grid_size, 5 + num_classes]
            y_true = [bbox_true_13, bbox_true_26, bbox_true_52]  三个list
            anchors: yolo模型对应的anchors
            num_classes: 类别数量
            ignore_thresh: 小于该阈值的box我们认为没有物体
        Returns
        -------
            loss: 每个batch的平均损失值
            accuracy
         Note
        ---------
           anchors: [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
        """
        loss = 0
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = [416.0, 416.0]

        #yolo_out[1] =[None,h,w,anchors,labels]=[None,26,26,3,85]  ,tf.shape(yolo_output[l])[1:3]=[26,26]        则得到的grid_shapes[[26,26],[26,26],[26,26]],注意这是list不是二维矩阵
        grid_shapes = [tf.cast(tf.shape(yolo_output[l])[1:3], tf.float32) for l in range(3)]

        for index in range(3):       #index对应的是第几个预测层
            # 只有负责预测ground truth box的grid对应的为1, 才计算相对应的loss
            # object_mask的shape为[batch_size, grid_size, grid_size, 3, 1]

            #y_true的格式[layers,None,h,w,anchors,labels]  其中预测层个数layers=3（这是一个list），anchors个数=3， h=26,w=26,labels=[cx,cy,w,h,obj,p1,p2,...,p80]  其中[cx,cy,w,h]都是相对于输入416*416后归一化到[0,1]的数据
            object_mask = y_true[index][..., 4:5]                             #取出当前预测层对应标签中是否为目标的分量，[None h,w,anchors,obj]=[None,26,16,3,1]
            class_probs = y_true[index][..., 5:]                              #取出当前预测层对应标签中目标类别的分量，[None h,w,anchors,p]=[None,26,16,3,80]

            grid, predictions, pred_xy, pred_wh = self.yolo_head(yolo_output[index], anchors[anchor_mask[index]], num_classes, input_shape, training = True)
            # pred_box的shape为[batch, box_num, 4]
            pred_box = tf.concat([pred_xy, pred_wh], axis = -1)                               #预测框[None 26 26 3,4]


           #我们认为网络输出的x，y是相对于网格各自格点的偏移量，则要计算损失的时候也需要将我们的label转换为偏移量，即raw_true_xy
            raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid         #y_true是标签，记录的x,y归一化后相对于全图的位置，这里乘以特征图的尺寸对应到特征图上，减去grid得到相对于grid的偏移量
            object_mask_bool = tf.cast(object_mask, dtype = tf.bool)                       #取出是目标




            #我们认为网络预测的w,h是相对于先验框一个e指数的比例，即 exp(w)*aw/input_shape，则计算损失的时候也需要将w,h的标签转换为相同的意义，即反解公式 ：w=log(w_real  *input_shape/aw)       其中w_real是标签指定的目标宽度
            raw_true_wh = tf.log(tf.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1], 0), tf.ones_like(y_true[index][..., 2:4]), y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1]))
            # 该系数是用来调整box坐标loss的系数
            box_loss_scale = 2 - y_true[index][..., 2:3] * y_true[index][..., 3:4]         #w*h 面积越大，scale值越小，对损失惩罚力度越小？

            ignore_mask = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)
            def loop_body(internal_index, ignore_mask):
                # true_box的shape为[box_num, 4]
                true_box = tf.boolean_mask(y_true[index][internal_index, ..., 0:4], object_mask_bool[internal_index, ..., 0])   #根据object_mask_bool中为true的位置，提取出所有的box, 输出维度为[boxes_num 4]
                iou = self.box_iou(pred_box[internal_index], true_box)                                                          #一副图中有n个框，拿这n个框与网格中所有位置预测的框计算IOU,输出的IOU为[h w anchors num_box]个值
                # 计算每个true_box对应的预测的iou最大的box
                best_iou = tf.reduce_max(iou, axis = -1)                                                                        #[h w anchors]       对于网格中某个网格而言，它的预测框与所有的真实的目标框都计算过IOU,这里只取最大的IOU，作为网格的IOU
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))              #在指定internal_index缩影位置写入 0，如果最大的IOU都小于设定阈值，则说明网格中的其中的那个网格对所用的真实的框的预测都不好（换句话说所有的真实的框都与这个网格无关），要忽略
                return internal_index + 1, ignore_mask

            #lambda表达式用于定义简单的函数：例如p = lambda x,y:x+y  计算p(4,5)=9,其中x,y是表达式的变量，冒号后面是表达式
            #while_loop用法：即loop参数先传入cond 判断条件是否成立，成立之后，把 loop参数传入body 执行操作， 然后返回 操作后的 loop 参数，即loop参数已被更新，再把更新后的参数传入cond, 依次循环，直到不满足条件
            # loop = []
            # while cond(loop):
            #     loop = body(loop)
            #这里 cond用lambda表达式来表示，tf.shape(yolo_output[0])[0]对应的是[None 26 26 255]中None所在的维度，也就是内部索引 internal_index其实是对每个样本的索引遍历
            #loop_body是要循环执行的函数，
            #[0,ignore_mask] 是循环中要用到的变量，0是internal_mask的初始值，ignore_mask是内部要循环使用的变量
            _, ignore_mask = tf.while_loop(lambda internal_index, ignore_mask : internal_index < tf.shape(yolo_output[0])[0], loop_body, [0, ignore_mask])

            #测试用
            # loop_body(0, ignore_mask)

            ignore_mask = ignore_mask.stack()                 #[None 13 13 3 ]
            ignore_mask = tf.expand_dims(ignore_mask, axis = -1)
            # 计算四个部分的loss
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels = raw_true_xy, logits = predictions[..., 0:2])    #[None h w anchors 2]
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) * ignore_mask
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels =  class_probs, logits = predictions[..., 5:])

            #除以batch值得到平均损失

            xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)                            #使用reduce_sum ，axis=None所有的维度数据求和  得到一个数据
            wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            class_loss = tf.reduce_sum(class_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)

            loss += xy_loss + wh_loss + confidence_loss + class_loss

        return loss
