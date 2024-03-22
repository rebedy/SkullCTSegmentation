import os
from os.path import join
import datetime
from glob import glob
import numpy as np
import csv

import torch
import torch.optim as optim                     # optimizers e.g. gradient descent, ADAM, etc.
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchgeometry.losses import DiceLoss as dice_loss

from args import parse_args
from models import get_model
import utils

global_step, global_eval_step = 0, 0


class CTSeg_BatchSliceDatasetV2(Dataset):

    def __init__(self, vtiPath, num_slices=5, seed=42):
        """
        Where the initial logic happens like transform.
        Preprocessing dataset happens here.
        Data downloading, reading, etc.
        """
        # Read Image, Make Image And MAsk Tensor
        imageData = utils.ReadVTI(vtiPath)
        imageArray = utils.ConvertVtkToNumpy(imageData)
        maskArray = utils.GenerateMaskImage(imageData, args.mask)

        self.image = torch.from_numpy(imageArray).float()
        self.mask = torch.from_numpy(maskArray).long()
        self.num_slices = num_slices
        self.seed = seed

    def __len__(self):  # return count of sample we have
        """ Trun over the size/length of dataset, the total number of samples."""
        return self.image.shape[0] - 4

    def __getitem__(self, index):
        indices = list(range(2, len(self.image) - 2))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        idx = indices[index]

        pre_idx = int(np.floor(self.num_slices / 2))  # == 5/2
        post_idx = self.num_slices - pre_idx  # == 5-2

        image_slices = self.image[idx - pre_idx: idx + post_idx]
        mask_slices = self.mask[idx]

        sample = {'image': image_slices, 'mask': mask_slices}

        return sample


class CTSeg_ValDataset(Dataset):
    def __init__(self, vtiPath, num_slices=5):
        """
        Where the initial logic happens like transform.
        Preprocessing dataset happens here.
        Data downloading, reading, etc.
        ! Same as CTSeg_BatchSliceDataset except this returns vti spacing values.
        """
        # ### TODO | DY : This is for the sake of when there is more than one Patient Batch.

        # Read Image, Make Image And MAsk Tensor
        imageData = utils.ReadVTI(vtiPath)
        imageArray = utils.ConvertVtkToNumpy(imageData)
        maskArray = utils.GenerateMaskImage(imageData, args.mask)

        self.image = torch.from_numpy(imageArray).float()
        self.mask = torch.from_numpy(maskArray).long()
        self.num_slices = num_slices

    def __len__(self):  # return count of sample we have
        """ Trun over the size/length of dataset, the total number of samples."""
        return self.image.shape[0] - 4

    def __getitem__(self, index):
        indices = list(range(2, len(self.image) - 2))
        idx = indices[index]
        pre_idx = int(np.floor(self.num_slices / 2))   # == 5/2
        post_idx = self.num_slices - pre_idx           # == 5-2

        image_slices = self.image[idx - pre_idx: idx + post_idx]
        mask_slices = self.mask[idx]
        sample = {'image': image_slices, 'mask': mask_slices}

        return sample


def Trainer(train_paths, SLICE_BATCH, device, opti, net, criterion, tb_summary_writer):
    global global_step
    # ### ! Training Loop ! ###
    tr_pati_loss = 0
    for p, vtiPath in enumerate(train_paths):
        tr_pati_num = os.path.basename(vtiPath)

        # ### ! * Training Dataset Load !
        CTSeg_slices = CTSeg_BatchSliceDatasetV2(vtiPath)
        ct_depth = len(CTSeg_slices)
        # ### TODO | DY : You cannot true the shuffle arg. here because of indexing.(See custom_dataset.py)
        train_loader = DataLoader(CTSeg_slices, batch_size=SLICE_BATCH, shuffle=False, num_workers=0)

        net.train()

        tr_loss = 0.0
        for batch in train_loader:
            input_tensor = batch['image'].to(device)
            gt_tensor = batch['mask'].to(device)

            opti.zero_grad()                # ### zero the gradient buffers; 변화도 버퍼를 0으로 설정
            pred = net(input_tensor)        # Output
            loss = criterion(pred, gt_tensor)
            loss.backward()                 # ### Back Propagation
            opti.step()                     # ### Does the update!!!

            tr_loss += loss.item()

            tb_summary_writer.add_scalar('0_Global_Step/Train_Loss', loss.item(), global_step)
            global_step += 1

        train_loss = tr_loss / ct_depth     # average loss of total batch of each patient
        tr_pati_loss += train_loss          # sum of average batch loss for patients

        print("▻  {}th Training Iteration | Patient {} | Patient Training Loss(mean) {}".format(str(p + 1), tr_pati_num, np.round(train_loss, 4)))

    return tr_pati_loss


def Validation(valid_paths, SLICE_BATCH, device, net, criterion, scheduler, tb_summary_writer):
    global global_eval_step
    # ### ! Validation Loop ! ###
    val_pati_loss = 0.0
    for v, val_patient in enumerate(valid_paths):
        val_num = os.path.basename(val_patient)

        # ### * Validation Dataset Load !
        val_CTSeg_slices = CTSeg_ValDataset(val_patient)
        val_ct_depth = len(val_CTSeg_slices)  # the number of batch
        valid_loader = DataLoader(val_CTSeg_slices, batch_size=SLICE_BATCH, shuffle=False, num_workers=0)

        net.eval()

        v_loss = 0.0  # ep_loss
        for v_idx, val_batch in enumerate(valid_loader):
            # ? if v_idx < 2 or v_idx > val_ct_depth-3 : continue
            val_imgs, val_true_masks = val_batch['image'], val_batch['mask']
            val_imgs = val_imgs.to(device)
            val_true_masks = val_true_masks.to(device)

            # ### ! * Actual Validation
            with torch.no_grad():
                val_pred = net(val_imgs)
                val_loss = criterion(val_pred, val_true_masks)

            scheduler.step(val_loss)
            v_loss += val_loss.item()

            tb_summary_writer.add_scalar('0_Global_Step/Validation_Loss', val_loss.item(), global_eval_step)
            global_eval_step += 1

        # ! --------------------------------------------------------------------------- !

        # ### *  Mean validation score for tot. batch of one valid_patient.
        valid_loss = v_loss / val_ct_depth
        val_pati_loss += valid_loss

        print("▻  {}th Validation | Patient {} | Loss(mean) {}".format(str(v + 1), val_num, np.round(valid_loss, 4)))

    return val_pati_loss


if __name__ == "__main__":

    print("\n")
    print("┌───────────────────────┐")
    print("│ CT Segmentaiton Start │")
    print("└───────────────────────┘\n")
    start = datetime.datetime.now()
    print("     @ Starts at %s" % str(start))

    # ### ! Args Parsing and Setting ! ###
    TODAY = str(datetime.date.today().strftime("%y%m%d"))
    NOW = str(datetime.datetime.now().strftime("_%Hh%Mm"))

    args = parse_args()

    # ### TODO | DY : Change directory and file name if transfer learning is required.

    # ### ! NETWORK and HYPER-PARAM! ###
    # ### * Device Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ### * Neural Network !!!
    net = get_model(args.model, args)
    net.to(device=device)  # Intialize Model

    # If Pretrained model exists
    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained, map_location=device))

    opti = optim.Adam(net.parameters(), lr=args.lr)  # weight_decay=LD)
    # criterion = nn.CrossEntropyLoss() ### => if net.n_classes > 1:
    criterion = dice_loss()  # (scale=2)
    try:
        log_criterion = criterion.__name__
    except AttributeError:
        log_criterion = criterion.__class__.__name__
    scheduler = lr_scheduler.ReduceLROnPlateau(opti, 'min' if args.out_dim > 1 else 'max', patience=2)

    dataLogStr = "-"
    for datapath in args.dir:
        dataLogStr += os.path.split(datapath)[-1] + "-"

    LOG = join(args.logdir, args.project_tag + 'logs-' + TODAY + NOW + dataLogStr + str(net.__class__.__name__) + '-' + str(log_criterion))
    os.makedirs(LOG, exist_ok=True)

    tb_summary_writer = SummaryWriter(log_dir=LOG, comment=f'LR_{args.lr}_BS_{args.slice_batch}')

    # ### ! Loading Dataset ! ###
    train_paths, valid_paths = [], []
    for dataPath in args.dir:

        train_paths += glob(os.path.join(dataPath, "train", "*.vti"))
        valid_paths += glob(os.path.join(dataPath, "valid", "*.vti"))

        # Filter
        metaData = csv.DictReader(open(join(dataPath, "meta.csv")))

        # Filter Dataset
        metaWrong = []
        for meta in metaData:
            for maskname in args.mask:
                if not meta[maskname] == "TRUE":
                    metaWrong.append(meta["path"])
                    break
        for wrong in metaWrong:
            if join(dataPath, "train", wrong) in train_paths:
                train_paths.remove(join(dataPath, "train", wrong))
            elif join(dataPath, "valid", wrong) in valid_paths:
                valid_paths.remove(join(dataPath, "valid", wrong))
            else:
                print(">>> Wrong mask could be in the dataset list. Please check the 'meta.csv' file.")

    np.random.seed(args.seed)
    np.random.shuffle(train_paths)
    np.random.shuffle(valid_paths)
    print('     >>> LOG : ', LOG)
    print('     >>> len(train_paths) : ', len(train_paths))
    print('     >>> len(valid_paths) : ', len(valid_paths))

    n_train = len(train_paths)
    n_valid = len(valid_paths)

    if n_train == 0:
        print("No Train Data ")
        exit()
    if n_valid == 0:
        n_valid = 1  # For Error handling

    # ### ! EPOCH Start ! ###
    for ep in range(args.epochs):

        # Train
        start_epoch = datetime.datetime.now()
        print("\n\n=============== [Epoch %d] " % (ep + 1), str(start_epoch), ' ===============\n')
        tr_pati_loss = Trainer(train_paths, args.slice_batch, device, opti, net, criterion, tb_summary_writer)

        checkpoint = {
            "model_state_dict": net.state_dict(),
            "args": args,
            "global_step": global_step,
            "current_epoch": ep,
            "optimizer_state_dict": opti.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.log, f'checkpoints_ep{ep + 1}.tar'))
        train_itr_done = datetime.datetime.now() - start_epoch
        print("\n   >>> %d th epoch training DONE! %s\n" % (ep + 1, str(train_itr_done)))
        print("-----------------------------------------------------------------------")

        # Validation
        start_validation = datetime.datetime.now()
        val_pati_loss = Validation(valid_paths, args.slice_batch, device, net, criterion, scheduler, tb_summary_writer)
        valid_itr_done = datetime.datetime.now() - start_validation
        print("\n   >>> %d th epoch validation DONE! %s\n" % (ep + 1, str(valid_itr_done)))
        print("-----------------------------------------------------------------------")

        tb_summary_writer.close()

        train_loss = tr_pati_loss / n_train  # len(train_paths)
        valid_loss = val_pati_loss / n_valid
        print('\n>>> *mean train loss : ', train_loss)
        print('>>> *mean_val_cross_entropy : ', valid_loss)

        tb_summary_writer.add_scalars('1_Each_Epoch/Loss', {
            'Train_Loss': train_loss,
            'Validation_Loss': valid_loss
        }, ep + 1)

        epoch_done = datetime.datetime.now() - start_epoch
        print('\n>>> %d th Epoch Done! It took for  %s' % (ep + 1, str(epoch_done)))
        print("=======================================================================")

    print('\n\n>>> Whole process took for  %s' % (str(datetime.datetime.now() - start)))
    print(datetime.datetime.now())
    print("=================================Done!======================================\n")
