import torch.nn as nn
from torchvision.models import (
    resnet101, ResNet101_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    resnext101_32x8d, ResNeXt101_32X8D_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    inception_v3, Inception_V3_Weights
)


def initialize_model_resnet(num_classes, activation_name, WEIGHTS = "pretrained"):
    activation = activation_name if isinstance(activation_name, nn.Module) else activation_function[activation_name]

    if WEIGHTS == "pretrained":
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
    else:
        model = resnet101(weights=None)
    # Replace the fully connected (fc) head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_classes),
        activation_name
    )
    return model


def initialize_model_mobilenet(num_classes, activation_name, WEIGHTS = "pretrained"):
    activation = activation_name if isinstance(activation_name, nn.Module) else activation_function[activation_name]
    
    if WEIGHTS == "pretrained":
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
    else:
        model = mobilenet_v3_large(weights=None)

    # Replace the fully connected (fc) head
    model.classifier[-1] = nn.Sequential(
        nn.Linear(model.classifier[-1].in_features, num_classes),
        activation
    )
    return model


def initialize_model_resnext(num_classes, activation_name, WEIGHTS = "pretrained"):

    activation = activation_name if isinstance(activation_name, nn.Module) else activation_function[activation_name]

    if WEIGHTS == "pretrained":
        weights = ResNeXt101_32X8D_Weights.DEFAULT
        model = resnext101_32x8d(weights=weights)
    else:
        model = resnext101_32x8d(weights=None)

    # Replace the fully connected (fc) head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_classes),
        activation
    )
    return model


def initialize_model_efficientnet(num_classes, activation_name, WEIGHTS = "pretrained"):
    activation = activation_name if isinstance(activation_name, nn.Module) else activation_function[activation_name]

    if WEIGHTS == "pretrained":
        weights = EfficientNet_V2_L_Weights.DEFAULT
        model = efficientnet_v2_l(weights=weights)
    else:
        model = efficientnet_v2_l(weights=None)

    # Replace the fully connected (fc) head
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, num_classes),
        activation
    )
    return model


def initialize_model_inceptionv3(num_classes, activation_name, WEIGHTS = "pretrained"):
    activation = activation_name if isinstance(activation_name, nn.Module) else activation_function[activation_name]

    if WEIGHTS == "pretrained":
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
    else:
        model = inception_v3(weights=None, aux_logits=True)

    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_classes),
        activation
    )

    if model.aux_logits:
        model.AuxLogits.fc = nn.Sequential(
            nn.Linear(model.AuxLogits.fc.in_features, num_classes),
            activation
        )
    
    return model