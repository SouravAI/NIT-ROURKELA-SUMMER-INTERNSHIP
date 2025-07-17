import torch.nn as nn
import torchvision.models as models

def get_model(model_name="mobilenetv3_small", num_classes=5, pretrained=False):
    if model_name == "mobilenetv3_small":
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=pretrained)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes

    elif model_name == "efficientnetv2_s":
        model = models.efficientnet_v2_s(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    return model
if __name__ == "__main__":
    import torch
    model = get_model("mobilenetv3_small", num_classes=5, pretrained=False)
    x = torch.randn(2, 1, 128, 63)  # Your mel shapes
    x = x.repeat(1, 3, 1, 1)        # Repeat channel to make it 3-channel
    out = model(x)
    print("Output shape:", out.shape)  # Should be (2, 5)
