from .mobilenetV2 import MobileNetV2
from .mobilenetV2BNX import MobileNetV2BNX
from .mobilenetV2Unet import MobileNetV2Unet
from .mobilenetV2BNXUnet import MobileNetV2BNXUnet
from .unet2D import UNet


def get_model(name, args):

    if name == "mobilev2":
        return MobileNetV2(n_channels=args.in_dim,
                           n_classes=args.out_dim,
                           input_size=args.input_size,
                           width_mult=args.width_mult)

    elif name == "mobilev2bnx":
        return MobileNetV2BNX(n_channels=args.in_dim,
                              n_classes=args.out_dim,
                              input_size=args.input_size,
                              width_mult=args.width_mult)

    elif name == "mobilev2unet":
        return MobileNetV2Unet(n_channels=args.in_dim,
                               n_classes=args.out_dim,)  # , pre_trained=args.pretrained_model)

    elif name == "mobilev2bnxunet":
        return MobileNetV2BNXUnet(n_channels=args.in_dim,
                                  n_classes=args.out_dim)

    elif name == "unet":
        return UNet(n_channels=args.in_dim, n_classes=args.out_dim)  # , bilinear=True)

    else:
        raise ValueError("Model not found")
