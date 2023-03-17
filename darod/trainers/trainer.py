import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ..layers.frcnn_layers import RoIDelta
from ..metrics import mAP
from ..utils import train_utils, log_utils


class Trainer:
    """
    Class to train DAROD model.
    """
    def __init__(self, config, model, labels, experiment_name, backbone,
                 backup_dir="/home/nxf67149/Documents/codes/DAROD/saves/"):
        """

        :param config: configuration dictionary with training settings
        :param model: the model to train
        :param labels (list): class labels
        :param experiment_name: name of the experiment
        :param backbone: backbone to use (DAROD or vision based one)
        :param backup_dir: directory to save logs
        """
        self.config = config
        self.eval_every = self.config["training"]["eval_every"]
        self.lr = self.config["training"]["lr"]
        optimizer = self.config["training"]['optimizer']
        self.scheduler = self.config["training"]["scheduler"]
        momentum = self.config["training"]["momentum"]
        self.variances_boxes = self.config["fastrcnn"]["variances_boxes"]
        self.total_labels = self.config["data"]["total_labels"]
        self.frcnn_num_pred = self.config["fastrcnn"]["frcnn_num_pred"]
        self.box_nms_score = self.config["fastrcnn"]['box_nms_score']
        self.box_nms_iou = self.config["fastrcnn"]['box_nms_iou']
        #
        decay_step = self.config["training"]["scheduler_step"]
        if self.scheduler:
            self.lr = ExponentialDecay(self.lr, decay_rate=0.9, staircase=True,
                                       decay_steps=(config["training"]["num_steps_epoch"] / config["training"][
                                           "batch_size"]) * decay_step)

        if optimizer == "SGD":
            self.optimizer = tf.optimizers.SGD(learning_rate=self.lr, momentum=momentum)
        elif optimizer == "WSGD":
            self.optimizer = tfa.optimizers.SGDW(learning_rate=self.lr, momentum=momentum, weight_decay=0.0005)
        elif optimizer == "adam":
            self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        elif optimizer == "adad":
            self.optimizer = tf.optimizers.Adadelta(learning_rate=1.0)
        elif optimizer == "adag":
            self.optimizer = tf.optimizers.Adagrad(learning_rate=self.lr)
        else:
            raise NotImplemented("Not supported optimizer {}".format(optimizer))
        #
        self.experience_name = experiment_name
        self.backbone = backbone
        #
        self.labels = labels
        self.n_epochs = self.config["training"]["epochs"]
        self.model = model

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.backup_dir = os.path.join(backup_dir, experiment_name)
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        with open(os.path.join(self.backup_dir, 'config.json'), 'w') as file:
            json.dump(config, file, indent=2)
        # Tensorboard setting
        self.summary_writer = tf.summary.create_file_writer(self.backup_dir)

        # Checkpoint manager
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model, step=self.global_step)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.backup_dir, max_to_keep=5)
        self.ckpt.restore(self.manager.latest_checkpoint)

        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            self.global_step.assign(self.ckpt.step.numpy())

    def init_metrics(self):
        """
        Initialise metrics for training
        :return:
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_rpn_reg_loss = tf.keras.metrics.Mean(name="train_rpn_reg_loss")
        self.train_rpn_cls_loss = tf.keras.metrics.Mean(name="train_rpn_cls_loss")
        self.train_frcnn_reg_loss = tf.keras.metrics.Mean(name="train_frcnn_reg_loss")
        self.train_frcnn_cls_loss = tf.keras.metrics.Mean(name="train_frcnn_cls_loss")

        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_rpn_reg_loss = tf.keras.metrics.Mean(name="val_rpn_reg_loss")
        self.val_rpn_cls_loss = tf.keras.metrics.Mean(name="val_rpn_cls_loss")
        self.val_frcnn_reg_loss = tf.keras.metrics.Mean(name="val_frcnn_reg_loss")
        self.val_frcnn_cls_loss = tf.keras.metrics.Mean(name="val_frcnn_cls_loss")

    def reset_all_metrics(self):
        """
        Reset all metrics
        :return:
        """
        self.train_loss.reset_states()
        self.train_rpn_reg_loss.reset_states()
        self.train_rpn_cls_loss.reset_states()
        self.train_frcnn_reg_loss.reset_states()
        self.train_frcnn_cls_loss.reset_states()
        self.val_loss.reset_states()
        self.val_rpn_reg_loss.reset_states()
        self.val_rpn_cls_loss.reset_states()
        self.val_frcnn_reg_loss.reset_states()
        self.val_frcnn_cls_loss.reset_states()

    def update_train_metrics(self, loss, rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss):
        """
        Update train metrics
        :param loss: total loss
        :param rpn_reg_loss: rpn regression loss
        :param rpn_cls_loss: rpn classification loss
        :param frcnn_reg_loss: fast rcnn regression loss
        :param frcnn_cls_loss: fast rcnn classification loss
        :return:
        """
        self.train_loss(loss)
        self.train_rpn_reg_loss(rpn_reg_loss)
        self.train_rpn_cls_loss(rpn_cls_loss)
        self.train_frcnn_reg_loss(frcnn_reg_loss)
        self.train_frcnn_cls_loss(frcnn_cls_loss)

    def update_val_metrics(self, loss, rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss):
        """
        Update val metrics
        :param loss: total loss
        :param rpn_reg_loss: rpn regression loss
        :param rpn_cls_loss: rpn classification loss
        :param frcnn_reg_loss: fast rcnn regression loss
        :param frcnn_cls_loss: fast rcnn classification loss
        :return:
        """
        self.val_loss(loss)
        self.val_rpn_reg_loss(rpn_reg_loss)
        self.val_rpn_cls_loss(rpn_cls_loss)
        self.val_frcnn_reg_loss(frcnn_reg_loss)
        self.val_frcnn_cls_loss(frcnn_cls_loss)

    def train_step(self, input_data, anchors, epoch):
        """
        Performs one training step (forward pass + optimisation)
        :param input_data: RD spectrum
        :param anchors: anchors
        :param epoch: epoch number
        :return:
        """
        spectrums, gt_boxes, gt_labels, is_same_seq, _, _ = input_data

        # Take only valid idxs
        valid_idxs = tf.where(is_same_seq == 1)
        spectrums, gt_boxes, gt_labels = tf.gather_nd(spectrums, valid_idxs), tf.gather_nd(gt_boxes,
                                                                                           valid_idxs), tf.gather_nd(
            gt_labels, valid_idxs)

        if spectrums.shape[0] != 0:

            # Get RPN labels
            bbox_deltas, bbox_labels = train_utils.calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels,
                                                                                self.config)

            # Train step
            with tf.GradientTape() as tape:
                rpn_cls_pred, rpn_delta_pred, frcnn_cls_pred, frcnn_reg_pred, roi_bboxes, _ = self.model(spectrums,
                                                                                                         training=True)

                frcnn_reg_actuals, frcnn_cls_actuals = RoIDelta(self.config)([roi_bboxes, gt_boxes, gt_labels])
                rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss = train_utils.darod_loss(rpn_cls_pred,
                                                                                                   rpn_delta_pred,
                                                                                                   frcnn_cls_pred,
                                                                                                   frcnn_reg_pred,
                                                                                                   bbox_labels,
                                                                                                   bbox_deltas,
                                                                                                   frcnn_reg_actuals,
                                                                                                   frcnn_cls_actuals)

            losses = rpn_reg_loss + rpn_cls_loss + frcnn_reg_loss + frcnn_cls_loss
            self.update_train_metrics(losses, rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss)

            gradients = tape.gradient([rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss],
                                      self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if self.scheduler:
                cur_lr = self.optimizer.lr(self.global_step)

            print("===> ite(epoch) {}({})/{}({})  <==> train loss {} <==> rpn_reg_loss {} <==> rpn_cls_loss {} \ "
                  "<==> frcnn_reg_loss {} <==> frcnn_cls_loss {}".format(self.global_step.value().numpy(), epoch,
                                                                         self.total_ite, self.n_epochs, losses.numpy(),
                                                                         rpn_reg_loss.numpy(), rpn_cls_loss.numpy(),
                                                                         frcnn_reg_loss.numpy(),
                                                                         frcnn_cls_loss.numpy()))

            with self.summary_writer.as_default():
                tf.summary.scalar("train_loss/total_loss", losses, step=self.global_step)
                tf.summary.scalar("train_loss/rpn_reg_loss", rpn_reg_loss, step=self.global_step)
                tf.summary.scalar("train_loss/rpn_cls_loss", rpn_cls_loss, step=self.global_step)
                tf.summary.scalar("train_loss/frcnn_reg_loss", frcnn_reg_loss, step=self.global_step)
                tf.summary.scalar("train_loss/frcnn_cls_loss", frcnn_cls_loss, step=self.global_step)
                if not self.scheduler:
                    tf.summary.scalar("lr", self.optimizer.learning_rate.numpy(), step=self.global_step)
                else:
                    tf.summary.scalar("lr", cur_lr.numpy(), step=self.global_step)

            if np.isnan(losses):
                raise ValueError("Get NaN at iteration ", self.ite)
            self.global_step.assign_add(1)

    def val_step(self, input_data, anchors):
        """
        Performs one validation step
        :param input_data: RD spectrum
        :param anchors: anchors
        :return: ground truth boxes, labels and predictions
        """
        spectrums, gt_boxes, gt_labels, is_same_seq, _, _ = input_data

        # Take only valid idxs
        valid_idxs = tf.where(is_same_seq == 1)
        spectrums, gt_boxes, gt_labels = tf.gather_nd(spectrums, valid_idxs), tf.gather_nd(gt_boxes,
                                                                                           valid_idxs), tf.gather_nd(
            gt_labels, valid_idxs)

        if spectrums.shape[0] != 0:
            # Get RPN labels
            bbox_deltas, bbox_labels = train_utils.calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels,
                                                                                self.config)
            rpn_cls_pred, rpn_delta_pred, frcnn_cls_pred, frcnn_reg_pred, roi_bboxes, decoder_output = self.model(
                spectrums, training=True)

            if self.config["fastrcnn"]["reg_loss"] == "sl1":
                # Smooth L1-loss
                frcnn_reg_actuals, frcnn_cls_actuals = RoIDelta(self.config)([roi_bboxes, gt_boxes, gt_labels])
                rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss = train_utils.darod_loss(rpn_cls_pred,
                                                                                                   rpn_delta_pred,
                                                                                                   frcnn_cls_pred,
                                                                                                   frcnn_reg_pred,
                                                                                                   bbox_labels,
                                                                                                   bbox_deltas,
                                                                                                   frcnn_reg_actuals,
                                                                                                   frcnn_cls_actuals)
            elif self.config["fastrcnn"]["reg_loss"] == "giou":
                # Generalized IoU loss
                expanded_roi_bboxes, expanded_gt_boxes, frcnn_cls_actuals = RoIDelta(self.config)(
                    [roi_bboxes, gt_boxes, gt_labels])
                rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss = train_utils.darod_loss_giou(rpn_cls_pred,
                                                                                                        rpn_delta_pred,
                                                                                                        frcnn_cls_pred,
                                                                                                        frcnn_reg_pred,
                                                                                                        bbox_labels,
                                                                                                        bbox_deltas,
                                                                                                        expanded_gt_boxes,
                                                                                                        frcnn_cls_actuals,
                                                                                                        expanded_roi_bboxes)

            losses = rpn_reg_loss + rpn_cls_loss + frcnn_reg_loss + frcnn_cls_loss
            self.update_val_metrics(losses, rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss)
            return gt_boxes, gt_labels, decoder_output
        else:
            return gt_boxes, gt_labels, None

    def train(self, anchors, train_dataset, val_dataset):
        """
        Train the model
        :param anchors: anchors
        :param train_dataset: train dataset
        :param val_dataset: validation dataset
        :return:
        """
        best_loss = np.inf
        self.ite = 0
        self.total_ite = int(
            (self.config["training"]["num_steps_epoch"] / self.config["training"]["batch_size"]) * self.n_epochs)
        self.init_metrics()

        start_epoch = int(self.global_step.numpy() / (
                    self.config["training"]["num_steps_epoch"] / self.config["training"]["batch_size"]))

        for epoch in range(start_epoch, self.n_epochs):

            # Reset metrics
            self.reset_all_metrics()
            # Perform one train step
            for input_data in train_dataset:
                self.train_step(input_data, anchors, epoch)

            # Testing phase
            if (epoch % self.eval_every == 0 and epoch != 0) or epoch == self.n_epochs - 1:
                tp_dict = mAP.init_tp_dict(len(self.labels) - 1, iou_threshold=[0.5])

            # Validation phase
            for input_data in val_dataset:
                gt_boxes, gt_labels, decoder_output = self.val_step(input_data, anchors)

                if decoder_output is not None and (
                        epoch % self.eval_every == 0 and epoch != 0) or epoch == self.n_epochs - 1:
                    pred_boxes, pred_labels, pred_scores = decoder_output
                    pred_labels = pred_labels - 1
                    gt_labels = gt_labels - 1
                    for batch_id in range(pred_boxes.shape[0]):
                        tp_dict = mAP.accumulate_tp_fp(pred_boxes.numpy()[batch_id], pred_labels.numpy()[batch_id],
                                                       pred_scores.numpy()[batch_id], gt_boxes.numpy()[batch_id],
                                                       gt_labels.numpy()[batch_id], tp_dict,
                                                       iou_thresholds=[0.5])

            if (epoch % self.eval_every == 0 and epoch != 0) or epoch == self.n_epochs - 1:
                ap_dict = mAP.AP(tp_dict, n_classes=len(self.labels) - 1)
                log_utils.tensorboard_val_stats(self.summary_writer, ap_dict, self.labels[1:], self.global_step)
                if self.val_loss.result().numpy() < best_loss:
                    best_loss = self.val_loss.result().numpy()
                    self.model.save_weights(os.path.join(self.backup_dir, "best-model.h5"))
                    self.config["training"]["best_epoch"] = epoch

            # Log metrics
            with self.summary_writer.as_default():
                tf.summary.scalar("val_loss/total_loss", self.val_loss.result().numpy(), step=self.global_step)
                tf.summary.scalar("val_loss/rpn_reg_loss", self.val_rpn_reg_loss.result().numpy(),
                                  step=self.global_step)
                tf.summary.scalar("val_loss/rpn_cls_loss", self.val_rpn_cls_loss.result().numpy(),
                                  step=self.global_step)
                tf.summary.scalar("val_loss/frcnn_reg_loss", self.val_frcnn_reg_loss.result().numpy(),
                                  step=self.global_step)
                tf.summary.scalar("val_loss/frcnn_cls_loss", self.val_frcnn_cls_loss.result().numpy(),
                                  step=self.global_step)

            print("<=============== Validation ===============>")
            print("===> epoch {}/{}  <==> val loss {} <==> val rpn_reg_loss {} <==> val rpn_cls_loss {}"
                  " <==> val frcnn_reg_loss {} <==> val frcnn_cls_loss {}".format(
                epoch, self.n_epochs, self.val_loss.result().numpy(), self.val_rpn_reg_loss.result().numpy(),
                self.val_rpn_cls_loss.result().numpy(), self.val_frcnn_reg_loss.result().numpy(),
                self.val_frcnn_cls_loss.result().numpy()))
            print("<==========================================>")

            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
