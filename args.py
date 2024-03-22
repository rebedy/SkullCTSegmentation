import argparse


def parse_args():
    """Parse input arguments."""
    desc = ('CT Segmentation Project\n')

    parser = argparse.ArgumentParser(description=desc)
    # ### ! [General] ! ###
    parser.add_argument('--project_tag', type=str, default="CTSeg_")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--in_dim', type=int, default=5)
    parser.add_argument('--out_dim', type=int, default=3)
    parser.add_argument('--slice_batch', type=int, default=4)

    parser.add_argument('--dir', type=str, nargs="+", metavar='DIR',
                        default=["//192.168.0.xxx/Data/",
                                 "//192.168.0.xxx/Data1/",],
                        help='The source dataset directory.')

    parser.add_argument('--model', type=str, default="mobilenet",
                        choices=["mobilenet", "mobilenetbnx", "mobilenetunet", "mobilenetbnxunet", "unet"])
    parser.add_argument("--datapath", nargs="+", type=str, default="/192.168.0.xxx/Data/")
    parser.add_argument("--pthfile", type=str, default="checkpoint.pth")
    parser.add_argument("--ckpt", type=str, default="checkpoint.tar")
    parser.add_argument('--output', type=str, default="/192.168.0.xxx/dcm2singlevti/")
    parser.add_argument('--log', type=str, default="meta.csv")
    parser.add_argument('--mask', type=str, nargs="+", default=["mx", "mn"])
    parser.add_argument("--logdir", default="./runs")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pretrained", type=str, default=None)
    # ### ! [MISC] ! ###

    args = parser.parse_args()
    return args
