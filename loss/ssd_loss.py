
import tensorflow as tf


class ssd_loss:

    def __init__(self, num_classes, alpha=1.0, neg_pos_ratios=3.0,background_label_id=0, negatives_for_hard=100.0) -> None:
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratios
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard
    
    def smooth_l1_loss(self, y_true, y_pred):

        absolute_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), sq_loss, absolute_loss - 0.5)
        return tf.math.reduce_sum(l1_loss, -1)
    
    def softmax_loss(self, y_ture, y_pred):
        y_pred = tf.maximum(y_pred, 1e-15)
        softmax_loss = -tf.math.reduce_sum(y_ture * tf.math.log(y_pred), axis=-1)
        return softmax_loss
    
    def compute_loss(self, y_true, y_pred):

        batch = tf.shape(y_true)[0]
        num_boxes = tf.cast(tf.shape(y_true)[1], tf.float32)

        
        # Conference loss
        conf_loss = self.softmax_loss(y_true[:, :, 4:-8], y_pred[:, :, 4:-8])
        # Location loss
        loc_loss = self.smooth_l1_loss(y_true[:, :, :4], y_pred[:, :, :4])
        # Positive loss
        pos_loc_loss = tf.math.reduce_sum(loc_loss * y_true[:, :, -8], axis=1)
        pos_conf_loss = tf.math.reduce_sum(conf_loss * y_true[:, :, -8], axis=1)
        
        # Positive sample
        num_pos = tf.math.reduce_sum(y_true[:, :, -8], axis=-1)

        # Negitave sample
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos)

        pos_num_neg_mask = tf.math.greater(num_neg, 0)
        has_min = tf.cast(tf.math.reduce_any(pos_num_neg_mask), tf.float32)
        num_neg = tf.concat(axis=0, values=[num_neg, [(1 - has_min) * self.negatives_for_hard]])

        num_neg_batch = tf.math.reduce_mean(tf.boolean_mask(num_neg, tf.math.greater(num_neg, 0)))
        num_neg_batch = tf.cast(num_neg_batch, tf.int32)
        
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes
        
        max_confs = tf.math.reduce_max(y_pred[:, :, confs_start:confs_end], axis=2)

        _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -8]), k=num_neg_batch)

        batch_idx = tf.expand_dims(tf.range(0, batch), 1) # (batch, 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))

        full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(num_boxes, tf.int32) + tf.reshape(indices, [-1]))

        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss, [batch, num_neg_batch])
        neg_conf_loss = tf.math.reduce_sum(neg_conf_loss, axis=1)

        # Normalization
        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        total_loss = tf.math.reduce_sum(pos_conf_loss) + tf.math.reduce_sum(neg_conf_loss)
        total_loss /= tf.math.reduce_sum(num_pos)
        total_loss += tf.math.reduce_sum(self.alpha * pos_loc_loss) / tf.math.reduce_sum(num_pos)
        return total_loss
        