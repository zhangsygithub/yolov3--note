import os
from model import config
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict




class Reader:
    def __init__(self, mode, data_dir, anchors_path, num_classes, tfrecord_num = 1, input_shape = 416, max_boxes = 20):
        """
        Introduction
        ------------
            构造函数
        Parameters
        ----------
            data_dir: 文件路径
            mode: 数据集模式
            anchors: 数据集聚类得到的anchor
            num_classes: 数据集图片类别数量
            input_shape: 图像输入模型的大小
            max_boxes: 每张图片最大的box数量
            jitter: 随机长宽比系数
            hue: 调整hsv颜色空间系数
            sat: 调整饱和度系数
            cont: 调整对比度系数
            bri: 调整亮度系数
        """
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.max_boxes = max_boxes
        self.mode = mode
        self.annotations_file = {'train' : config.train_annotations_file, 'val' : config.val_annotations_file}
        self.data_file = {'train': config.train_data_file, 'val': config.val_data_file}
        self.anchors_path = anchors_path
        self.anchors = self._get_anchors()
        self.num_classes = num_classes
        file_pattern = self.data_dir + "/*" + self.mode + '.tfrecords'
        self.TfrecordFile = tf.gfile.Glob(file_pattern)
        self.class_names = self._get_class(config.classes_path)
        if len(self.TfrecordFile) == 0:
            self.convert_to_tfrecord(self.data_dir, tfrecord_num)                     #

    def _get_anchors(self):
        """
        Introduction
        ------------
            获取anchors
        Returns
        -------
            anchors: anchor数组
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_class(self, classes_path):
        """
        Introduction
        ------------
            获取类别名字
        Returns
        -------
            class_names: coco数据集类别对应的名字
        """
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def Preprocess_true_boxes(self, true_boxes):
        """
        Introduction
        ------------
            对训练数据的ground truth box进行预处理
        Parameters
        ----------
            true_boxes: ground truth box 形状为[boxes, 5], x_min, y_min, x_max, y_max, class_id
        Note
        ---------
           anchors: [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]

        """
        num_layers = len(self.anchors) // 3                            #每个层使用3中anchor
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')             #true_boxes是tf.float32类型，转换为numpy类型
        input_shape = np.array([self.input_shape, self.input_shape], dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2. #中心点坐标cx
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]         #w,h

        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]           #将cx,cy按照网络输入尺寸416*416归一化   [num_boxes 5]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]


        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]     #三个预测层上网格的尺度  416/32=13   416/16=26  416/8=52  [[13,13],[26,26],[52,52]]

        #生成中间尺度的标签  [26,26,3,85]  其中3表示该层上生成3种anchor  ,85=5+classes对于coco数据集合而言
        #后面for l in range(num_layers) 相当于循环生成list列表 得到【3,26,26,3,85】的张量作为标签，其中第一个3代表3个预测层，第二个3表示每个层有三种anchor
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes), dtype='float32') for l in range(num_layers)]


        # 这里扩充维度是为了后面应用广播计算每个图中所有box的anchor互相之间的iou
        #anchors: [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]

        #为了给图像中的目标框，选择与其IOU最大的anchor,采用的方式是，将anchor中心固定在（IOU坐标系的）原点，不动，在坐标轴两边对称分布，这里的坐标系不是图像坐标系，而是单独建立的与图像无关的坐标系
        #而图中目标框的位置确实相对于图像坐标系的，针对某一个目标框，怎么让其中心对齐到（IOU坐标系的）原点呢，其实也很简单，就是将框的坐标减去框中心点的坐标就可以了
        #但是这里采用了一种更简单的方式，就是直接框的宽度除以2，添加正负号分别当做x方向的框，letf,right对应的坐标，直接就对应到了原点，6666


        anchors = np.expand_dims(self.anchors, 0)    #本来是[9,2],每一行对应一中尺度的anchor,扩展为[1,9,2]的张量
        anchors_max = anchors / 2.
        anchors_min = -anchors_max
        # 因为之前对box做了padding(一副图中框的个数不足20个的时候则padding 0,补足到20个box), 因此需要去除全0行
        valid_mask = boxes_wh[..., 0] > 0      #boxes_wh的shape为[num_boxes 2]
        wh = boxes_wh[valid_mask]              #筛选之后的shape为[n 2]
        # 为了应用广播扩充维度
        wh = np.expand_dims(wh, -2)
        # wh 的shape为[box_num, 1, 2]
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        #anchor 和box的维度一样吗？
        intersect_min = np.maximum(boxes_min, anchors_min)      #boxes_min的shape为 [box_num, 1, 2]，anchors_min的shape为[1,9,2],这样就可以利用传播机制做运算了
        intersect_max = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)    #iou的shape为[box_num 9]

        #一个框找到与之最大的iou的anchor后，根据该anchor所在的层数，第几种anchor类型，来对输出标签对应层数、对应位置、对应anchor种类打标签
        # anchors: [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
        #例如某一个某个目标框[x,y,w,h] =[100,200,30,61] 很显然和第4个anchor的IOU最大，而第4个anchor=[30,61]对应的预测层为第2层，anchor的种类是第一种
        #第二层特征图尺寸为26*26  则对应到特征图上的位置[26*100/416,26*200/416]=[6.25 13.5]=[6,13]的位置，
        #则标签y_true[1][6][13][0][0:4]=[[100,200,30,61]/416.0]
        #类别向量采用one_hot的形式
        #整个打标签的方式和yolo_v1的方式一样

        # 找出和ground truth box的iou最大的anchor box, 然后将对应不同比例的负责该ground turth box 的位置置为ground truth box坐标
        best_anchor = np.argmax(iou, axis = -1)     #iou的shape为[box_num 9 ],则best_anchor的shape为 [boxes_num],best_anchor中存放的第一个对象是第一个框对应的最大iou的anchor 的索引值
        for t, n in enumerate(best_anchor):         #enumerate是按照axis=0的维度来枚举对象， 这里n代表第t个框对应最大iou的索引值（argmax）
            for l in range(num_layers):
                if n in anchor_mask[l]:             #因为argmax的索引值范围[0,8]，mask的范围也是，这样就可以判断了
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)     #例如n是属于第1组mask的第2个值,(从0开始索引)， 则k=2,表示的意思是当前框与第一组anchor中的第2个anchor的iou最大，所以在标签中将第一个预测层的第2个anchor所在的位置打上标签
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1.
                    y_true[l][j, i, k, 5 + c] = 1.
        return y_true[0], y_true[1], y_true[2]



    def read_annotations(self):
        """
        Introduction
        ------------
            读取COCO数据集图片路径和对应的标注
        Parameters
        ----------
            data_file: 文件路径
        """
        image_data = []
        boxes_data = []
        name_box_id = defaultdict(list)
        with open(self.annotations_file[self.mode], encoding='utf-8') as file:
            data = json.load(file)
            annotations = data['annotations']
            for ant in annotations:
                id = ant['image_id']
                name = os.path.join(self.data_file[self.mode], '%012d.jpg' % id)  #文件路径
                cat = ant['category_id']

                #这里之所以这样判断，是因为coco标的name编号不连续，从./model_data/labelmap_coco.prototxt中可以看出来 其中的name=12,26,29,30等10个是没有对应的编号的
                #这样判断后得到的编号就是0-79一共80个类别了
                if cat >= 1 and cat <= 11:
                    cat = cat - 1
                elif cat >= 13 and cat <= 25:
                    cat = cat - 2
                elif cat >= 27 and cat <= 28:
                    cat = cat - 3
                elif cat >= 31 and cat <= 44:
                    cat = cat - 5
                elif cat >= 46 and cat <= 65:
                    cat = cat - 6
                elif cat == 67:
                    cat = cat - 7
                elif cat == 70:
                    cat = cat - 9
                elif cat >= 72 and cat <= 82:
                    cat = cat - 10
                elif cat >= 84 and cat <= 90:
                    cat = cat - 11
                name_box_id[name].append([ant['bbox'], cat])  #name为路径作为键，值为[x,y,w,h,class]

            for key in name_box_id.keys():
                boxes = []
                image_data.append(key)         #key是图片路径
                box_infos = name_box_id[key]   #值为[x,y,w,h,class]将其转换为[x_min,y_min,x_max,y_max]的形式
                for info in box_infos:
                    x_min = info[0][0]
                    y_min = info[0][1]
                    x_max = x_min + info[0][2]
                    y_max = y_min + info[0][3]
                    boxes.append(np.array([x_min, y_min, x_max, y_max, info[1]]))
                boxes_data.append(np.array(boxes))

        return image_data, boxes_data


    def convert_to_tfrecord(self, tfrecord_path, num_tfrecords):
        """
        Introduction
        ------------
            将图片和boxes数据存储为tfRecord
        Parameters
        ----------
            tfrecord_path: tfrecord文件存储路径
            num_tfrecords: 分成多少个tfrecord
        """
        image_data, boxes_data = self.read_annotations()     #image_data是图片路径，boxes_data是[x1,y1,x2,y2,class]
        images_num = int(len(image_data) / num_tfrecords)
        for index_records in range(num_tfrecords):
            output_file = os.path.join(tfrecord_path, str(index_records) + '_' + self.mode + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):
                    with tf.gfile.FastGFile(image_data[index], 'rb') as file:
                        image = file.read()
                        xmin, xmax, ymin, ymax, label = [], [], [], [], []
                        for box in boxes_data[index]:
                            xmin.append(box[0])
                            ymin.append(box[1])
                            xmax.append(box[2])
                            ymax.append(box[3])
                            label.append(box[4])
                        example = tf.train.Example(features = tf.train.Features(
                            feature = {
                                'image/encoded' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                                'image/object/bbox/xmin' : tf.train.Feature(float_list = tf.train.FloatList(value = xmin)),
                                'image/object/bbox/xmax': tf.train.Feature(float_list = tf.train.FloatList(value = xmax)),
                                'image/object/bbox/ymin': tf.train.Feature(float_list = tf.train.FloatList(value = ymin)),
                                'image/object/bbox/ymax': tf.train.Feature(float_list = tf.train.FloatList(value = ymax)),
                                'image/object/bbox/label': tf.train.Feature(float_list = tf.train.FloatList(value = label)),
                            }
                        ))
                        record_writer.write(example.SerializeToString())
                        if index % 1000 == 0:
                            print('Processed {} of {} images'.format(index + 1, len(image_data)))


    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据，数据集合是通过tfrecord建立起来的，则数据集合中的每个元素是tfrecord中的序列号的example
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/encoded' : tf.FixedLenFeature([], dtype = tf.string),
                'image/object/bbox/xmin' : tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(dtype = tf.float32),
                'image/object/bbox/label': tf.VarLenFeature(dtype = tf.float32)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, axis = 0)       #扩展一个维度用于下面的conta
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, axis = 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, axis = 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, axis = 0)
        label = tf.expand_dims(features['image/object/bbox/label'].values, axis = 0)
        bbox = tf.concat(axis = 0, values = [xmin, ymin, xmax, ymax, label])           #xmin  ymin等是一个list按照行排列，concat是axis=0,则会有5行，每一列是一个样本的标签
        bbox = tf.transpose(bbox, [1, 0])                                              #装置一下，每一行变成一个样本的标签  [5,1]-->[1,5]

        # 图像预处理包括缩放 padding 随机翻转，输出416*416的图，同时box也做相应的变换，每个图带20个目标框
        image, bbox = self.Preprocess(image, bbox)

        #将一幅图中的box打包成形式同网络三个yolo预测层的输出形式的张量，用作标签 ，标签是list中的3个元素 元素是[13 13 3 85] [26,26,3 85]  [52 52 3 85]
        bbox_true_13, bbox_true_26, bbox_true_52 = tf.py_func(self.Preprocess_true_boxes, [bbox], [tf.float32, tf.float32, tf.float32])
                                                                                                                            #Py_func是将一个python函数打包成一个tensonflow的op,因为普通的python函数是不可直接输入tensorflow的变量的，
                                                                                                                            # 使用的方式是将这个函数打包成python的op，在运行的时候就可以调用了
                                                                      #self.Preprocess_true_boxes 是待打包的函数， [bbox]该函数输入变量列表     [tf.float32, tf.float32, tf.float32] 是输出到tensorflow中的变量的类型

        return image, bbox, bbox_true_13, bbox_true_26, bbox_true_52

    def Preprocess(self, image, bbox):
        """
        Introduction
        ------------
            对图片进行预处理，增强数据集
        Parameters
        ----------
            image: tensorflow解析的图片
            bbox: 图片中对应的box坐标
        """
        image_width, image_high = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)           #将w,h转换为浮点数
        #网络输入尺寸
        input_width = tf.cast(self.input_shape, tf.float32)
        input_high = tf.cast(self.input_shape, tf.float32)

        #输入图像按照长边按照416对待，短边等比例缩放，例如输入图像为  832*600的图像，按照这一原则，  new_height=600*416/832=300, new_width =832*416/832=416
        new_high = image_high * tf.minimum(input_width / image_width, input_high / image_high)
        new_width = image_width * tf.minimum(input_width / image_width, input_high / image_high)
        image = tf.image.resize_images(image, [tf.cast(new_high, tf.int32), tf.cast(new_width, tf.int32)], method=tf.image.ResizeMethod.BICUBIC)
        # 将图片按照固定长宽比进行padding缩放，四周按0填充（预处理最后用opencv来做）
        dx = (input_width - new_width) / 2        #x方向单边padding的像素个数
        dy = (input_high - new_high) / 2
        new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))

        #生成同样大小全为1的矩阵，作为一个mask,再次以相同方式padding ，得到图像区域全1，填充区域全0的mask
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32), tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))

        #将mask反向，乘以128再加到原来的图像上，这样达到周围padding区域全为128的灰度区域
        image_color_padded = (1 - image_ones_padded) * 128
        image = image_color_padded + new_image
        # 矫正bbox坐标
        xmin, ymin, xmax, ymax, label = tf.split(value = bbox, num_or_size_splits=5, axis = 1)   # box是每一行一个样本，这里按照行，分离出x1,y1,x2,y2,class
        xmin = xmin * new_width / image_width + dx                                               #将坐标对应到缩放后再padding的图上的坐标
        xmax = xmax * new_width / image_width + dx
        ymin = ymin * new_high / image_high + dy
        ymax = ymax * new_high / image_high + dy
        bbox = tf.concat([xmin, ymin, xmax, ymax, label], 1)                                   #重新concat
        if self.mode == 'train':
            # 随机左右翻转图片
            def _flip_left_right_boxes(boxes):
                xmin, ymin, xmax, ymax, label = tf.split(value = boxes, num_or_size_splits = 5, axis = 1)
                flipped_xmin = tf.subtract(input_width, xmax)
                flipped_xmax = tf.subtract(input_width, xmin)
                flipped_boxes = tf.concat([flipped_xmin, ymin, flipped_xmax, ymax, label], 1)
                return flipped_boxes
            #根据随机数是否大于0.5来决定是否左右翻转图片，如果翻转，则图片和box标签都要翻转，图片翻转调用tf.image自带的函数，box翻转使用自定义的函数
            flip_left_right = tf.greater(tf.random_uniform([], dtype = tf.float32, minval = 0, maxval = 1), 0.5)    #随机生成一个符合正太分部的数，这个数大于0.5，则flip_left_right为True,否则为false
            image = tf.cond(flip_left_right, lambda : tf.image.flip_left_right(image), lambda : image)              #tf.cond是一个条件函数，如果flip_left_right=true,则返回第一lambda函数作用后的结果，否则返回第二个lambda函数作用后的结果
            bbox = tf.cond(flip_left_right, lambda: _flip_left_right_boxes(bbox), lambda: bbox)


        # 将图片归一化到0和1之间
        image = image / 255.
        image = tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)
        bbox = tf.clip_by_value(bbox, clip_value_min = 0, clip_value_max = input_width - 1)

        #如果一副图像中目标框的个数大于20个，则只取钱20个框，如果不足20个框，则在补成20个框，补充框的[x1,y1,x2,y2,class]都为0
        bbox = tf.cond(tf.greater(tf.shape(bbox)[0], config.max_boxes), lambda: bbox[:config.max_boxes], lambda: tf.pad(bbox, paddings = [[0, config.max_boxes - tf.shape(bbox)[0]], [0, 0]], mode ='CONSTANT'))
        return image, bbox


    def build_dataset(self, batch_size):
        """
        Introduction
        ------------
            建立数据集dataset
        Parameters
        ----------
            batch_size: batch大小
        Return
        ------
            dataset: 返回tensorflow的dataset
        """
        dataset = tf.data.TFRecordDataset(filenames = self.TfrecordFile)       #使用tfrecord文件初始化数据集
        dataset = dataset.map(self.parser, num_parallel_calls = 10)            #调用parser自定义的预处理函数，从tfrecord中，每次读取10个样本进行并行预处理
        if self.mode == 'train':

           # from tensorflow.contrib.data import *
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(100))   #使用tf.contrib.data.shuffle_and_repeat(100)变换方法对数据集合进行打乱,输入的100是buffersize
            dataset = dataset.batch(batch_size).prefetch(batch_size)
        else:
            dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
        return dataset
