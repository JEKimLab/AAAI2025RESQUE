import sys
import json
import torch

sys.path.append("/home/lsangar/FilterSelectImageClassification/helpers")
from save_train_misc import output_to_std_and_file

sys.path.append("/home/lsangar/FilterSelectImageClassification/models")
from VisT import ViT
from googlenet import GoogleNet
from mobilenet_v2 import MobileNetV2 
from resnet import ResNet, resnet_cfg
from vgg import VGG, make_layers, vgg_cfg

from resnet_in64_1 import ResNet_1, resnet_cfg_1
from resnet_in64_2 import ResNet_2, resnet_cfg_2
from resnet_in64_3 import ResNet_3, resnet_cfg_3
from resnet_in64_4 import ResNet_4, resnet_cfg_4

def create_model(model_type, model_layers, classes, save_to, vit_config_path=None):
    if model_type.lower() == "vgg":
        std_string = model_type.lower()
        model = VGG(make_layers(vgg_cfg[model_layers], batch_norm=True), classes)

    elif model_type.lower() == "resnet":
        std_string = model_type.lower()
        block = resnet_cfg[model_layers]["blocktype"]
        layer_cfg = resnet_cfg[model_layers]["layers"]
        model = ResNet(block, layer_cfg, classes)

    elif model_type.lower() == "resnet_in64_1":
        std_string = model_type.lower()
        block = resnet_cfg_1[model_layers]["blocktype"]
        layer_cfg = resnet_cfg_1[model_layers]["layers"]
        model = ResNet_1(block, layer_cfg, classes)

    elif model_type.lower() == "resnet_in64_2":
        std_string = model_type.lower()
        block = resnet_cfg_2[model_layers]["blocktype"]
        layer_cfg = resnet_cfg_2[model_layers]["layers"]
        model = ResNet_2(block, layer_cfg, classes)

    elif model_type.lower() == "resnet_in64_3":
        std_string = model_type.lower()
        block = resnet_cfg_3[model_layers]["blocktype"]
        layer_cfg = resnet_cfg_3[model_layers]["layers"]
        model = ResNet_3(block, layer_cfg, classes)

    elif model_type.lower() == "resnet_in64_4":
        std_string = model_type.lower()
        block = resnet_cfg_4[model_layers]["blocktype"]
        layer_cfg = resnet_cfg_4[model_layers]["layers"]
        model = ResNet_4(block, layer_cfg, classes)

    elif model_type.lower() == "googlenet":
        std_string = model_type.lower()
        model = GoogleNet(classes)

    elif model_type.lower() == "mobilenet":
        std_string = model_type.lower()
        model = MobileNetV2(classes)

    elif model_type.lower() == "vit":
        std_string = model_type.lower()

        with open(vit_config_path, "r") as file:
            vit_data = file.read()
        vit_data = json.loads(vit_data)

        image_size = vit_data["image_size"]
        patch_size = vit_data["patch_size"]
        dim = vit_data["dimensions"]
        depth = vit_data["depth"]
        heads = vit_data["attention_heads"]
        mlp_dim = vit_data["mlp_dimension"]
        dropout = vit_data["dropout"]
        emb_dropout = vit_data["embedding_dropout"]

        model = ViT(image_size=image_size, patch_size=patch_size, 
                    num_classes=classes, 
                    dim=dim, mlp_dim=mlp_dim,
                    depth=depth, heads=heads, 
                    dropout=dropout, emb_dropout=emb_dropout)

        std_string += f"\nimage_size: {image_size}"
        std_string += f"\npatch_size: {patch_size}"
        std_string += f"\ndimension: {dim}"
        std_string += f"\nmlp dimension: {mlp_dim}"
        std_string += f"\ndepth: {depth}"
        std_string += f"\nmlp heads: {heads}"
        std_string += f"\ndropout: {dropout}"
        std_string += f"\nembedding dropout: {emb_dropout}"


    std_string += "\n"
    std_string += str(model)
    output_to_std_and_file(save_to, "model_layers.txt", std_string)
    return model