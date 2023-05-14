import os
from argparse import ArgumentParser
import glob
import torch.utils.data as Data
import numpy as np
import torch

from Functions import generate_grid_unit, Dataset_epoch,  save_img, save_flow, transform_unit_flow_to_flow, load_4D_with_header, imgnorm
from miccai2021_model_lite import Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1, \
    Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2, Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit


parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='D:/ismi_data/Model/Stage/Lite_nlst_tryoutstagelvl3_600.pth',
                    help="Trained model path")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=4,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='D:/ismi_data/NLST/NLST_testdata/imagesTs/NLST_0111_0000.nii.gz',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='D:/ismi_data/NLST/NLST_testdata/imagesTs/NLST_0111_0001.nii.gz',
                    help="moving image")
parser.add_argument("--reg_input", type=float,
                    dest="reg_input", default=0.1,
                    help="Normalized smoothness regularization (within [0,1])")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='D:/ismi_data/NLST',
                    help="data path for test images")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving

if not os.path.isdir(savepath):
    os.mkdir(savepath)

start_channel = opt.start_channel
reg_input = opt.reg_input


def test():
    print("Current reg_input: ", str(reg_input))

    model_lvl1 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                                                         range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    model = Miccai2021_LDR_conditional_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=imgshape,
                                                                   range_flow=range_flow, model_lvl2=model_lvl2).cuda()


    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    names = sorted(glob.glob(opt.datapath +'/NLST_testdata/imagesTs'+ '/*.nii.gz'))

    test_generator = Data.DataLoader(Dataset_epoch(names, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    
    with torch.no_grad():
    
        for i, (X, Y) in enumerate(test_generator):
            X = X.cuda().float()
            Y = Y.cuda().float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            F_X_Y = model(X, Y, reg_code)
            X_Y = transform(Y, F_X_Y.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = F_X_Y.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)
            subj = i + 111

            #save_flow(F_X_Y_cpu, savepath + '/warpped_flow_full_' + 'reg' + str(subj) + str(reg_input) + '.nii.gz', header=header, affine=affine)
            save_img(X_Y, savepath + '/warpped_moving_full_' + 'reg' + str(subj) + str(reg_input) + '.nii.gz', header=header, affine=affine)

    print("Results saved to :", savepath)


if __name__ == '__main__':
    imgshape = (224, 192, 224)
    imgshape_4 = (224 / 4, 192 / 4, 224 / 4)
    imgshape_2 = (224 / 2, 192 / 2, 224 / 2)


    range_flow = 0.4
    test()
