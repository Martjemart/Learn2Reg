import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from sklearn.preprocessing import normalize
import torch.nn.functional as F
import tqdm

import numpy as np
import torch
import torch.utils.data as Data
import multiprocessing

from Functions import generate_grid, Dataset_epoch, Dataset_epoch_lvl3, Dataset_epoch_validation, Dataset_epoch_mask, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit, res_mask_2, res_mask_4
from miccai2021_model_lite import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, multi_resolution_NCC

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=111,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=111,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=60001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=1.,
                    help="Anti-fold loss: suggested range 1 to 10000")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=10,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=4,  # default:8, 7 for stage
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='C:/Users/Jelle/Documents/GitHub/NLST',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=3000,
                    help="Number of step to freeze the previous level")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

model_name = "Lite_nlst_tryout"
def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1

    return dice / num_count

#training first pyramid part
def train_lvl1():
    print("Training lvl1...")
    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape_4,
                                                                    range_flow=range_flow).cuda()

    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # NLST
    names = sorted(glob.glob(datapath +'/imagesTr'+ '/*.nii.gz'))

    grid_4 = generate_grid(imgshape_4)
    grid_4 = torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model_dir = './Model/Stage'
    os.makedirs('./Model', exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl1 + 1))

    #use mask loader
    training_generator = Data.DataLoader(Dataset_epoch_mask(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "./Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("./Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl1:
        for X, mask_0, Y, mask_1 in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y, reg_code)
            
            #fit mask to img shape
            mask_0_re = res_mask_4(mask_0).to(F_X_Y.device)
            mask_1_re = res_mask_4(mask_1).to(F_X_Y.device)

            #calculate loss for total img + masked img
            loss_multiNCC = loss_similarity(X_Y, Y_4x, mask_0_re, mask_1_re)

            # Normalize the NCC loss based on the value of 10 million
            normalized_loss_multiNCC = torch.where(loss_multiNCC > 10_000_000, torch.tensor(20.0), (loss_multiNCC / 10_000_000) * 19.0 + 1.0)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1)
            norm_vector[0, 1, 0, 0, 0] = (y - 1)
            norm_vector[0, 2, 0, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = normalized_loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), normalized_loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f} -reg_c "{5:.4f}"'.format(
                    step, loss.item(), normalized_loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl1_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl1_" + str(step) + '.npy', lossall)

            step += 1

            if step > iteration_lvl1:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl1.npy', lossall)

#train part two of the pyramid
def train_lvl2():
    print("Training lvl2...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()

    model_path = sorted(glob.glob("./Model/Stage/" + model_name + "stagelvl1_???.pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape_2,
                                                                    range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # NLST
    names = sorted(glob.glob(datapath +'/imagesTr'+ '/*.nii.gz'))

    grid_2 = generate_grid(imgshape_2)
    grid_2 = torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl2 + 1))

    training_generator = Data.DataLoader(Dataset_epoch_mask(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "./Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl2:
        for X, mask_0, Y, mask_1 in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y, reg_code)

            #fit mask to img shape
            mask_0_re = res_mask_2(mask_0).to(F_X_Y.device)
            mask_1_re = res_mask_2(mask_1).to(F_X_Y.device)
            #calculate loss for total img + masked img
            loss_multiNCC = loss_similarity(X_Y, Y_4x, mask_0_re, mask_1_re)

            # Normalize the NCC loss based on the value of 10 million
            normalized_loss_multiNCC = torch.where(loss_multiNCC > 10_000_000, torch.tensor(20.0), (loss_multiNCC / 10_000_000) * 19.0 + 1.0)


            F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0, 2, 3, 4, 1).clone())

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1)
            norm_vector[0, 1, 0, 0, 0] = (y - 1)
            norm_vector[0, 2, 0, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            smo_weight = reg_code * max_smooth
            loss = normalized_loss_multiNCC + antifold * loss_Jacobian + smo_weight * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), normalized_loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f} -reg_c "{5:.4f}"'.format(
                    step, loss.item(), normalized_loss_multiNCC.item(), loss_Jacobian.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl2_" + str(step) + '.npy', lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl2.npy', lossall)

def denormalization(deformation_field):
    B, C, D, H, W = deformation_field.size()     
    scale = torch.ones(deformation_field.size())
    scale[:, 0, :, :, :] = scale[:, 0, :, :, :] * (D - 1) / 2     
    scale[:, 1, :, :, :] = scale[:, 1, :, :, :] * (H - 1) / 2     
    scale[:, 2, :, :, :] = scale[:, 2, :, :, :] * (W - 1) / 2      
    deformation_conv =  deformation_field*scale.cuda()
    return deformation_conv

def trilinear_interpolate(im, x, y, z):
    im = im.cuda()

    dtype = torch.float

    B, C, D, H, W = im.size()
    x = x.type(dtype)
    y = y.type(dtype)
    z = z.type(dtype)

    x0 = torch.floor(x).type(torch.long)
    x1 = x0 + 1
    y0 = torch.floor(y).type(torch.long)
    y1 = y0 + 1
    z0 = torch.floor(z).type(torch.long)
    z1 = z0 + 1

    w0 = torch.clamp(z0, 0, W - 1)
    w1 = torch.clamp(z1, 0, W - 1)
    h0 = torch.clamp(y0, 0, H - 1)
    h1 = torch.clamp(y1, 0, H - 1)
    d0 = torch.clamp(x0, 0, D - 1)
    d1 = torch.clamp(x1, 0, D - 1)

    # image values of neighbors
    Ia = (im[:, :, d0, h0, w0]).to(im.device)
    Ib = (im[:, :, d1, h0, w0]).to(im.device)
    Ic = (im[:, :, d0, h1, w0]).to(im.device)
    Id = (im[:, :, d1, h1, w0]).to(im.device)
    Ie = (im[:, :, d0, h0, w1]).to(im.device)
    If = (im[:, :, d1, h0, w1]).to(im.device)
    Ig = (im[:, :, d0, h1, w1]).to(im.device)
    Ih = (im[:, :, d1, h1, w1]).to(im.device)

    # compute interpolation weights
    wa = ((d1.type(dtype) - x) * (h1.type(dtype) - y) * (w1.type(dtype) - z)).to(im.device)
    wb = ((d1.type(dtype) - x) * (h1.type(dtype) - y) * (z - w0.type(dtype))).to(im.device)
    wc = ((d1.type(dtype) - x) * (y - h0.type(dtype)) * (w1.type(dtype) - z)).to(im.device)
    wd = ((d1.type(dtype) - x) * (y - h0.type(dtype)) * (z - w0.type(dtype))).to(im.device)
    we = ((x - d0.type(dtype)) * (h1.type(dtype) - y) * (w1.type(dtype) - z)).to(im.device)
    wf = ((x - d0.type(dtype)) * (h1.type(dtype) - y) * (z - w0.type(dtype))).to(im.device)
    wg = ((x - d0.type(dtype)) * (y - h0.type(dtype)) * (w1.type(dtype) - z)).to(im.device)
    wh = ((x - d0.type(dtype)) * (y - h0.type(dtype)) * (z - w0.type(dtype))).to(im.device)

    I_dhw = Ia * wa + Ib * wb + Ic * wc + Id * wd + Ie * we + If * wf + Ig * wg + Ih * wh
    return I_dhw[0]

def landmarkDistance(moving, fixed, voxelSpacing):
    distance = (moving - fixed)
    distance[:,:, 0] *= voxelSpacing[:, 0].view(-1, 1) 
    distance[:,:, 1] *= voxelSpacing[:, 1].view(-1, 1)
    distance[:,:, 2] *= voxelSpacing[:, 2].view(-1, 1)

    dist = torch.sqrt((distance ** 2).sum(dim=2)).mean()
    return dist

def train_lvl3():
    print("Training lvl3...")
    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow,
                                                                         model_lvl1=model_lvl1).cuda()

    model_path = sorted(glob.glob("./Model/Stage/" + model_name + "stagelvl2_???.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = smoothloss

    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath +'/imagesTr'+ '/*.nii.gz'))

    # grid = generate_grid(imgshape)
    # grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    grid_unit = generate_grid_unit(imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration_lvl3 + 1))

    training_generator = Data.DataLoader(Dataset_epoch_lvl3(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "./Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("./Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl3:
        for X, mask_0, key_0, Y, mask_1, key_1, spacingA in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            mask_0 = mask_0.to(F_X_Y.device)
            mask_1 = mask_1.to(F_X_Y.device)

            # Inside the training loop
            loss_multiNCC = loss_similarity(X_Y, Y_4x, mask_0, mask_1)

            # Normalize the NCC loss based on the value of 10 million
            normalized_loss_multiNCC = torch.where(loss_multiNCC > 10_000_000, torch.tensor(20.0), (loss_multiNCC / 10_000_000) * 19.0 + 1.0)

            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = (z - 1)
            norm_vector[0, 1, 0, 0, 0] = (y - 1)
            norm_vector[0, 2, 0, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector)

            # Assuming F_X_Y is a deformation field and key_y is a set of coordinates in the 'Y' space
            # Normalize F_X_Y to be between -1 and 1 (assuming it's not already normalized)
            norm_tensor = torch.tensor([F_X_Y.shape[-1] / 2, F_X_Y.shape[-2] / 2, F_X_Y.shape[-3] / 2])
            norm_tensor = norm_tensor.view(1, 3, 1, 1, 1).to(F_X_Y.device)
            F_X_Y_normalized = F_X_Y.permute(0, 2, 3, 4, 1)
            F_X_Y_normalized = F_X_Y_normalized.float()

            deNormalized_conv = denormalization(F_X_Y)

            d = key_1[:,:,0].unsqueeze(-1)
            h = key_1[:,:,1].unsqueeze(-1)
            w = key_1[:,:,2].unsqueeze(-1)

            key_0 = key_0.cuda()
            key_1 = key_1.cuda()
            spacingA = spacingA.cuda()
            
            deformation_atMpositions = trilinear_interpolate(deNormalized_conv, d,h,w)

            deformation_atMpositions = torch.squeeze(deformation_atMpositions, 1)

            newLM = key_0 + deformation_atMpositions.permute(2,1,0)

            dist = landmarkDistance(newLM,key_1,spacingA)

            # Add keypoint_loss to the existing loss with a suitable weight
            smo_weight = reg_code * max_smooth
            #loss = loss_multiNCC + smo_weight * loss_regulation + keypoint_loss_weight * keypoint_loss

            loss = normalized_loss_multiNCC + smo_weight * loss_regulation + dist
            
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), normalized_loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f} -reg_c "{4:.4f}" -dist "{5:.4f}"'.format(
                    step, loss.item(), normalized_loss_multiNCC.item(), loss_regulation.item(),
                    reg_code[0].item(), dist.item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Put your validation code here
                # ---------------------------------------
                # NLST (Validation)
                names = sorted(glob.glob(datapath +'/imagesVal'+ '/*.nii.gz'))

                valid_generator = Data.DataLoader(Dataset_epoch(names, norm=True), batch_size=1,
                                         shuffle=False, num_workers=2)

                dice_total = []
                use_cuda = True
                device = torch.device("cuda" if use_cuda else "cpu")
                print("\nValiding...")
                for batch_idx, data in enumerate(valid_generator):
                    X, Y, X_label, Y_label = data[0].to(device), data[1].to(device), data[2].to(
                        device), data[3].to(device)

                    with torch.no_grad():
                        reg_code = torch.tensor([0.1], dtype=X.dtype, device=X.device).unsqueeze(dim=0)

                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)
                        X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit).cpu().numpy()[0, 0, :, :, :]
                        Y_label = Y_label.cpu().numpy()[0, 0, :, :, :]

                        dice_score = dice(np.floor(X_Y_label), np.floor(Y_label))
                        dice_total.append(dice_score)

                print("Dice mean: ", np.mean(dice_total))
                with open(log_dir, "a") as log:
                    log.write(str(step) + ":" + str(np.mean(dice_total)) + "\n")

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    imgshape = (224, 192, 224)
    imgshape_4 = (224 / 4, 192 / 4, 224 / 4)
    imgshape_2 = (224 / 2, 192 / 2, 224 / 2)

    torch.cuda.empty_cache()

    # Create and initalize log file
    if not os.path.isdir("./Log"):
        os.mkdir("./Log")

    log_dir = "./Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation Dice log for " + model_name[0:-1] + ":\n")

    range_flow = 0.4
    max_smooth = 10.
    start_t = datetime.now()
    train_lvl1()
    train_lvl2()
    train_lvl3()
    # time
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())
