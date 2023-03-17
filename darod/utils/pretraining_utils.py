import os

import tensorflow as tf


def load_from_imagenet(model_source, source_path, model_dest):
    """
    Load backbone weights pretrained on image net

    :param model_source: the pre-trained classification model on ImageNet dataset
    :param source_path: path to the weights of the source model
    :param model_dest: the model to initialize with
    :return: initialized model
    """
    model_source.load_weights(source_path)
    fake_input = tf.zeros((1, 256, 64, 3))
    _ = model_source(fake_input)
    for layer in model_dest.layers:
        if "rgg_block" in layer.name:
            # assert model_source.get_layer(layer.name).get_weights() != model_dest.get_layer(layer.name).get_weights()
            print("Initialize layer {} of the backbone with ImageNet weights...".format(layer.name))
            pretrained_weigths = model_source.get_layer(layer.name).get_weights()
            model_dest.get_layer(layer.name).set_weights(pretrained_weigths)
            # assert model_source.get_layer(layer.name).get_weights() == model_dest.get_layer(layer.name).get_weights()
            print("Layer {} initialized with ImageNet weights.".format(layer.name))
        else:
            continue
    print("Backbone successfully initialized!")
    return model_dest


def load_from_radar_ds(weights_path, model_dest):
    """
    Load a model trained on a other radar dataset (except classification and regression heads

    :param weights_path: path to the weights of the model
    :param model_dest: the model to initialize with
    :return: the intialized model
    """
    # Copy RPN weights before initialization
    weights_rpn = model_dest.get_layer("rpn_conv").get_weights()
    weights_cls = model_dest.get_layer("rpn_cls").get_weights()
    weights_reg = model_dest.get_layer("rpn_reg").get_weights()

    print("Try to initialiaze model using pretrained model on {} dataset...".format(weights_path.split("/")[-2]))
    weights_path = os.path.join(weights_path, "best-model.h5")
    model_dest.load_weights(weights_path, by_name=True, skip_mismatch=True)
    model_dest.get_layer("rpn_conv").set_weights(weights_rpn)
    model_dest.get_layer("rpn_cls").set_weights(weights_cls)
    model_dest.get_layer("rpn_reg").set_weights(weights_reg)
    print("Succesfully initialiazed model using pretrained model on {} dataset!".format(weights_path.split("/")[-3]))
    return model_dest
