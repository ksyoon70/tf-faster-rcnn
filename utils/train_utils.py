import tensorflow as tf
import math
from utils import bbox_utils

RPN = {
    "vgg16": {
        "img_size": 500,
        "feature_map_shape": 31,
        "anchor_ratios": [1., 2., 1./2.],
        "anchor_scales": [128, 256, 512],
    },
    "mobilenet_v2": {
        "img_size": 500,
        "feature_map_shape": 32,
        "anchor_ratios": [1., 2., 1./2.],
        "anchor_scales": [128, 256, 512],
    }
}

def get_hyper_params(backbone, **kwargs):
    """Generating hyper params in a dynamic way.
    inputs:
        **kwargs = any value could be updated in the hyper_params

    outputs:
        hyper_params = dictionary
    """
    hyper_params = RPN[backbone]
    hyper_params["pre_nms_topn"] = 6000
    hyper_params["train_nms_topn"] = 1500
    hyper_params["test_nms_topn"] = 300
    hyper_params["nms_iou_threshold"] = 0.7
    hyper_params["total_pos_bboxes"] = 128
    hyper_params["total_neg_bboxes"] = 128
    hyper_params["pooling_size"] = (7, 7)
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value
    #
    hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])
    return hyper_params

def get_step_size(total_items, batch_size):
    """Get step size for given total item size and batch size.
    inputs:
        total_items = number of total items
        batch_size = number of batch size during training or validation
    outputs:
        step_size = number of step size for model training
    """
    return math.ceil(total_items / batch_size)

def randomly_select_xyz_mask(mask, select_xyz):
    """Selecting x, y, z number of True elements for corresponding batch and replacing others to False
    inputs:
        mask = (batch_size, [m_bool_value])
        select_xyz = ([x_y_z_number_for_corresponding_batch])
            example = tf.constant([128, 50, 42], dtype=tf.int32)
    outputs:
        selected_valid_mask = (batch_size, [m_bool_value])
    """
    maxval = tf.reduce_max(select_xyz) * 10
    random_mask = tf.random.uniform(tf.shape(mask), minval=1, maxval=maxval, dtype=tf.int32)
    multiplied_mask = tf.cast(mask, tf.int32) * random_mask
    sorted_mask = tf.argsort(multiplied_mask, direction="DESCENDING")
    sorted_mask_indices = tf.argsort(sorted_mask)
    selected_mask = tf.less(sorted_mask_indices, tf.expand_dims(select_xyz, 1))
    return tf.logical_and(mask, selected_mask)

def faster_rcnn_generator(dataset, anchors, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            yield (img, gt_boxes, gt_labels, bbox_deltas, bbox_labels), ()

def rpn_generator(dataset, anchors, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            bbox_deltas, bbox_labels = calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params)
            yield img, (bbox_deltas, bbox_labels)

def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, hyper_params):
    """Generating one step data for training or inference.
    Batch operations supported.
    inputs:
        anchors = (total_anchors, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary

    outputs:
        bbox_deltas = (batch_size, total_anchors, [delta_y, delta_x, delta_h, delta_w])
        bbox_labels = (batch_size, feature_map_shape, feature_map_shape, anchor_count)
    """
    batch_size = tf.shape(gt_boxes)[0]
    feature_map_shape = hyper_params["feature_map_shape"]
    anchor_count = hyper_params["anchor_count"]
    total_pos_bboxes = hyper_params["total_pos_bboxes"]
    total_neg_bboxes = hyper_params["total_neg_bboxes"]
    variances = hyper_params["variances"]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(anchors, gt_boxes)
    # Get max index value for each row
    max_indices_each_row = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # Get max index value for each column
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    #tf.reduce_max 지정한 차원을 따라 최댓값을 계산
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    # 대소를 비교하여 true, false를 리턴한다.
    pos_mask = tf.greater(merged_iou_map, 0.7)
    # gt_labels가 -1 이 아니면 true를 반환하고 아니면 false를 반환한다. 즉 true , false의 array를 생성한다.
    valid_indices_cond = tf.not_equal(gt_labels, -1)
    # 생성된 true, false arrray를 가지고, true 인것에 대한 것만 가지고 ground true gt_labels 인덱스를 생성한다.
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    # gt_lables가 -1이 아닌 것에 대한 max가 되는 index만을 취한다.
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    #아래 코드는 현재 코드를  테스트 하기 위한 코드 이다.
    """
    gt_labels = tf.constant([1,2,-1,4])
    iou_map =  tf.constant([[0.8,0.7],[0.9,0.6],[0.5,0.1],[0.8,0.85]])
    max_indices_each_column = tf.argmax(iou_map, axis=1, output_type=tf.int32)

    valid_indices_cond = tf.not_equal(gt_labels, -1)
    #array([ True,  True, False,  True])
    valid_indices = tf.cast(tf.where(valid_indices_cond), tf.int32)
    # -1이 아닌것에 대한 인덱스만 뽑아낸다.
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    print(valid_max_indices.eval)
    by 윤경섭
    """
    #tf.stack(
    #values, axis=0, name='stack'
    #)
    """
    stack 예제
    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    print("tf.stack([x, y, z])",tf.stack([x, y, z]))
    
    결과
    
    tf.stack([x, y, z]) tf.Tensor(
        [[1 4]
         [2 5]
         [3 6]], shape=(3, 2), dtype=int32)

    tf.stack([x, y, z], axis=1)
    print("tf.stack([x, y, z], axis=1)",tf.stack([x, y, z], axis=1))
    결과
    tf.stack([x, y, z], axis=1) tf.Tensor(
        [[1 2 3]
         [4 5 6]], shape=(2, 3), dtype=int32)
    """
    # gt_labels 와 max_index를 stack으로 만들어 scatter_bbox_indices 를 생성한다.
    scatter_bbox_indices = tf.stack([valid_indices[..., 0], valid_max_indices], 1)
    """
    Creates a new tensor by applying sparse updates to individual values or slices within a tensor 
    (initially zero for numeric, empty for string) of the given shape according to indices. 
    This operator is the inverse of the tf.gather_nd operator which extracts values or slices from a given tensor.
    예
    indices = tf.constant([[4], [3], [1], [7]])
    updates = tf.constant([9, 10, 11, 12])
    shape = tf.constant([8])
    scatter = tf.scatter_nd(indices, updates, shape)
    결과
    [0, 11, 0, 10, 9, 0, 0, 12]
    """
    """
    tf.fill(dims, value, name=None)
    스칼라값으로 채워진 텐서를 생성합니다.
    이 연산은 dims shape의 텐서를 만들고 value로 값을 채웁니다.
    fill([2, 3], 9) ==> [[9, 9, 9]
                     [9, 9, 9]]
    """
    # tf.shape(valid_indices)[0] 는 valid_indices 텐서의 행의 크기를 구한다.
    # 그러므로 tf.fill((tf.shape(valid_indices)[0], ), True) 는 valid_indices 텐서의 행 크기의 [True, True,...] 행을 만든다.
    # iou 만큼의 행렬에서 valid_indices에 해당하는 곳에 True를 채운다. 나머지는 0
    max_pos_mask = tf.scatter_nd(scatter_bbox_indices, tf.fill((tf.shape(valid_indices)[0], ), True), tf.shape(pos_mask))
    pos_mask = tf.logical_or(pos_mask, max_pos_mask)
    pos_mask = randomly_select_xyz_mask(pos_mask, tf.constant([total_pos_bboxes], dtype=tf.int32))
    #
    pos_count = tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1)
    neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    #
    neg_mask = tf.logical_and(tf.less(merged_iou_map, 0.3), tf.logical_not(pos_mask))
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count)
    #
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_row, batch_dims=1)
    # Replace negative bboxes with zeros
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # Calculate delta values between anchors and ground truth bboxes
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances
    #
    # bbox_deltas = tf.reshape(bbox_deltas, (batch_size, feature_map_shape, feature_map_shape, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, feature_map_shape, feature_map_shape, anchor_count))
    #
    return bbox_deltas, bbox_labels

def frcnn_cls_loss(*args):
    """Calculating faster rcnn class loss value.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = CategoricalCrossentropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    loss_fn = tf.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    #
    cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    mask = tf.cast(cond, dtype=tf.float32)
    #
    conf_loss = tf.reduce_sum(mask * loss_for_all)
    total_boxes = tf.maximum(1.0, tf.reduce_sum(mask))
    return conf_loss / total_boxes

def rpn_cls_loss(*args):
    """Calculating rpn class loss value.
    Rpn actual class value should be 0 or 1.
    Because of this we only take into account non -1 values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = BinaryCrossentropy value
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    indices = tf.where(tf.not_equal(y_true, tf.constant(-1.0, dtype=tf.float32)))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy()
    return lf(target, output)

def reg_loss(*args):
    """Calculating rpn / faster rcnn regression loss value.
    Reg value should be different than zero for actual values.
    Because of this we only take into account non zero values.
    inputs:
        *args = could be (y_true, y_pred) or ((y_true, y_pred), )

    outputs:
        loss = Huber it's almost the same with the smooth L1 loss
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, 4))
    #
    loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.NONE)
    loss_for_all = loss_fn(y_true, y_pred)
    loss_for_all = tf.reduce_sum(loss_for_all, axis=-1)
    #
    pos_cond = tf.reduce_any(tf.not_equal(y_true, tf.constant(0.0)), axis=-1)
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    #
    loc_loss = tf.reduce_sum(pos_mask * loss_for_all)
    total_pos_bboxes = tf.maximum(1.0, tf.reduce_sum(pos_mask))
    return loc_loss / total_pos_bboxes
