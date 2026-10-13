import argparse
import logging
import math
import os
import shutil
import random
import sys
import copy
import pandas as pd
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import albumentations as A
from tqdm import tqdm
from PIL import Image
# from IPython import embed

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchsummary import summary

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from data.SuperResolution import Sen2VenusDataset
from data.util import bgr2ycbcr

# torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != "spawn":  
        # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group
    
def visualize_save_triplet_ir(x_sat, y_venus, model_output, idx = 0, fuse = True,only_IR = False, save = True, visualize = False, 
             cur_state = 5000,path = 'val_images/'):
    if fuse:
        x_sat_rgb, y_venus_rgb = x_sat[:3].cpu().detach().numpy().astype(np.float32),y_venus[:3].cpu().detach().numpy()
    else:
        x_sat_rgb, y_venus_rgb = x_sat[:3].numpy(),y_venus[:3].numpy()
    x_sat_rgb, y_venus_rgb = x_sat_rgb[::-1].transpose(1,2,0), y_venus_rgb[::-1].transpose(1,2,0)
#     model_output_rgb = np.array(model_output[0][4:7].cpu().detach())[::-1].transpose(1,2,0)
    image_processed = model_output.cpu()#*255#.permute(0, 2, 3, 1)#*255
    image_processed = image_processed.numpy().astype(np.float32)
    rgb = image_processed[:3,:,:]
    rgb = rgb[::-1].transpose(1,2,0)
    if visualize:
        fig, ax = plt.subplots(1,3, figsize=(18,6))
        ax[0].imshow(x_sat_rgb*3)
        ax[1].imshow(rgb*3)
        ax[2].imshow(y_venus_rgb*3)

        ax[0].set_title('IR patch sentinel ')
        ax[1].set_title('IR unet SR ')
        ax[2].set_title('IR patch venus ')

        plt.show()
    if save: 
        os.makedirs(path + cur_state, exist_ok=True)
        rgb = rgb = np.clip((rgb*3*255), 0,255).astype(np.uint8)
        x_sat_rgb = np.clip((x_sat_rgb*3*255), 0,255).astype(np.uint8)
        y_venus_rgb = np.clip((y_venus_rgb*3*255), 0,255).astype(np.uint8)
        im_x_sat_rgb = Image.fromarray(x_sat_rgb)
        im_x_sat_rgb.save(path+cur_state+'/IR_patch_sentinel_'+str(idx)+'.png')
        im_rgb = Image.fromarray(rgb)
        im_rgb.save(path+cur_state+'/IR_unet_SR_'+str(idx)+'.png')
        im_y_venus_rgb = Image.fromarray(y_venus_rgb)
        im_y_venus_rgb.save(path+cur_state+'/IR_patch_venus_'+str(idx)+'.png')    
    
def visualize_save_triplet_rgb(x_sat, y_venus, model_output, idx = 0, fuse = True, only_IR = False, save = True, visualize = False, 
             cur_state = 5000, path = 'val_images/'):             
    if fuse or x_sat.shape[0] == 8:
        x_sat_rgb, y_venus_rgb = x_sat[4:7].cpu().detach().numpy().astype(np.float32),y_venus[4:7].cpu().detach().numpy()
    else:
        x_sat_rgb, y_venus_rgb = x_sat[:3].numpy(),y_venus[:3].numpy()
    x_sat_rgb, y_venus_rgb = x_sat_rgb[::-1].transpose(1,2,0), y_venus_rgb[::-1].transpose(1,2,0)
#     model_output_rgb = np.array(model_output[0][4:7].cpu().detach())[::-1].transpose(1,2,0)
    image_processed = model_output.cpu()#*255#.permute(0, 2, 3, 1)#*255
    image_processed = image_processed.numpy().astype(np.float32)
    if fuse or image_processed.shape[0]==8:
        rgb = image_processed[4:7,:,:]
    else:
        rgb = image_processed[:3,:,:]
    rgb = rgb[::-1].transpose(1,2,0)
    if visualize:
        fig, ax = plt.subplots(1,3, figsize=(18,6))
        ax[0].imshow(x_sat_rgb*3)
        ax[1].imshow(rgb*3)
        ax[2].imshow(y_venus_rgb*3)

        ax[0].set_title('rgb patch sentinel ')
        ax[1].set_title('rgb unet SR ')
        ax[2].set_title('rgb patch venus ')

        plt.show()
    if save: 
        os.makedirs(path + cur_state, exist_ok=True)
        rgb = rgb = np.clip((rgb*3*255), 0,255).astype(np.uint8)
        x_sat_rgb = np.clip((x_sat_rgb*3*255), 0,255).astype(np.uint8)
        y_venus_rgb = np.clip((y_venus_rgb*3*255), 0,255).astype(np.uint8)
        im_x_sat_rgb = Image.fromarray(x_sat_rgb)
        im_x_sat_rgb.save(path+cur_state+'/RGB_patch_sentinel_'+str(idx)+'.png')
        im_rgb = Image.fromarray(rgb)
        im_rgb.save(path+cur_state+'/RGB_unet_SR_'+str(idx)+'.png')
        im_y_venus_rgb = Image.fromarray(y_venus_rgb)
        im_y_venus_rgb.save(path+cur_state+'/RGB_patch_venus_'+str(idx)+'.png')

def norm(img):
    return torch.tensor((img - 0.5) * 2.0,dtype=torch.float32)
    
def denorm(img):
    return torch.tensor(img / 2.0 + 0.5, dtype=torch.float32)

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="options/train/setting2x_sen2venus_l1.yml")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"  # none means disabled distributed training
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )#os.makedirs(dir_name, exist_ok=True)
            )

            log_path = "./log"

            # Remove existing file or directory named 'log'
#             if os.path.islink(log_path) or os.path.isfile(log_path):
#                 os.remove(log_path)
#             elif os.path.isdir(log_path):
#                 os.rmdir(log_path)

#             # Create the symbolic link
#             os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), log_path)
            if os.path.islink(log_path) or os.path.isfile(log_path):
                os.remove(log_path)
            elif os.path.isdir(log_path):
                shutil.rmtree(log_path)  # Recursively delete a directory tree

            # Create the symbolic link
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), log_path)

#             os.system("rm ./log")
#             os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")
# initialize datasets
    parameters = {
            "batch_size"    : opt['datasets']['train']['batch_size'],
            "n_epochs"      : 10,# not used actually
            "base_lr"       : 0.0005,
            "train_dataset" : "Sen2Venus",
            "valid_dataset" : "Sen2Venus",
            "train_csv"     : "train.csv",
            "valid_csv"     : "valid.csv",# true order is 4IR -> BGR, NIR
            "bands"         : ["B2", "B3", "B4", 'B5',"B6", 'B7',"B8", 'D8A'],#["B2", "B3", "B4", "B8"],
            "in_size"       : (256, 256),
            "out_size"      : (256, 256)
    }
#     root = 'sen2venus/sen2venus/npz'
    S2V_NPZ = os.environ.get('S2V_NPZ', 'data/sen2venus/npz')          # patches (npz)
    S2V_SPLITS = os.environ.get('S2V_SPLITS', 'data/sen2venus/splits')  # train.csv, valid.csv
    rooot = S2V_NPZ
    train_split = pd.read_csv(os.path.join(S2V_SPLITS, 'train.csv'))#f'sen2venus/sen2venus/splits/{parameters["train_csv"]}')
    valid_split = pd.read_csv(os.path.join(S2V_SPLITS, 'valid.csv'))
    valid_split = valid_split.iloc[:10000]# to boost validation process drastically
    train_transform = A.Compose(
        [A.RandomCrop(width=128, height=128, always_apply = True)],
        additional_targets={'image_high': 'image'}
    )
    val_transform = A.Compose(
        [A.CenterCrop(width=128, height=128,  always_apply = True)],
        additional_targets={'image_high': 'image'}
    )
    loaders_params = {
            #'phase':
            "n_workers":8,
            "root" : rooot,
            "train_split" : train_split,
            "valid_split" : valid_split,
            "batch_size"  : parameters['batch_size'],
            "lr"          : 'sentinel_10m',
            "hr"          : 'venus_5m_10m',
            "fuse"        : opt['datasets']['train']["fuse"],#False
            "only_IR"     : opt['datasets']['train']["only_IR"]
        }
    kwargs = {
            "lr"          : 'sentinel_10m',
            "hr"          : 'venus_5m_10m',
            "fuse"        : opt['datasets']['train']["fuse"],
            "only_IR"     : opt['datasets']['train']["only_IR"]
            }
    train_set = Sen2VenusDataset(root = rooot, dataframe = train_split, transform = None, **kwargs)
    val_set = Sen2VenusDataset(root = rooot, dataframe = valid_split, transform = None, **kwargs)
    print('csv sucessufully read and datasets compiled')
    # create dataloaders
    train_size = int(math.ceil(len(train_set) / loaders_params["batch_size"]))
    total_iters = int(opt["train"]["niter"])
    total_epochs = int(math.ceil(total_iters / train_size))
    dataset_ratio = 10 #mystery feature ?????????????
    if opt["dist"]:
        train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
        total_epochs = int(
            math.ceil(total_iters / (train_size * dataset_ratio))
        )
    else:
        train_sampler = None
    print('total num of epochs to train:',total_epochs)
    train_loader = create_dataloader(train_set,loaders_params , opt, train_sampler)#dataset_opt
    if opt["dist"]:
        val_sampler = DistIterSampler(val_set, world_size, rank, 1)
    else:
        val_sampler = None
    #
    val_loader = create_dataloader(val_set, loaders_params, opt, val_sampler)#dataset_opt
    
    model = create_model(opt) 
    device = model.device
    print('model created')

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        print('pretrained checkpoint will be loaded')
        model.resume_training(resume_state)  # handle optimizers and schedulers
        print('pretrained checkpoint sucessfully loaded')
    else:
        current_step = 0#5000
        start_epoch = 0#5000

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    best_ssim = 0.0
    best_ssim_iter = 0
    error = mp.Value('b', False)

    dev_rank = max(rank, 0)  # rank is -1 without distributed training
    psnr = PeakSignalNoiseRatio(data_range=1.0, reduction='sum', dim=(1,2,3)).to(dev_rank)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='sum').to(dev_rank)
    idx = 0
    # -------------------------------------------------------------------------
    # -------------------------正式开始训练，前面都是废话---------------------------
    # -------------------------------------------------------------------------
    for epoch in tqdm(range(start_epoch, total_epochs + 1)):
        logger.info("Starting new epoch!")
        print('training epoch num:', epoch)
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
        
            if current_step > total_iters:
                break

            LQ, GT = train_data#train_data["LQ"], train_data["GT"]  #  b 8 256 256; b 8 256 256
            LQ = norm(LQ)
            GT = norm(GT)

            # random timestep and state (noisy map) via SDE
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)  # t=batchsize，states [b 3 128 128]

            model.feed_data(states, LQ, GT)  # xt, mu, x0, 将加了噪声的LR图xt，LR以及GT输入改进的UNet进行去噪
            
            model.optimize_parameters(current_step, timesteps, sde)  # 优化UNet

            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
#                 logger.info('model saved after the epoch=',epoch)
                    print('model saved after the epoch=',epoch)
                print('Saving model saved on epoch=',epoch)
                model.save(current_step)
                model.save_training_state(epoch, current_step)
                
            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0:
                print('validation process started')
                avg_psnr = torch.tensor(0.0).to(dev_rank)
                avg_ssim = torch.tensor(0.0).to(dev_rank)
                cnt = torch.tensor(0).to(dev_rank) #if opt["dist"] else 0
                
                save_path = str(opt["path"]["experiments_root"]) + '/val_images/' + str(current_step)
                if rank <= 0:
                    util.mkdirs(save_path)
                
                for _, val_data in tqdm(enumerate(val_loader)):

                    LQ, GT = val_data
                    LQ = norm(LQ)
                    GT = norm(GT)
                    noisy_state = sde.noise_state(LQ)  # 在LR上加噪声，得到噪声LR图，噪声是随机生成的

                    # valid Predictor
                    model.feed_data(noisy_state, LQ, GT)
                    model.test(sde)
                    
                    avg_psnr += psnr(model.output, model.state_0)
                    avg_ssim += ssim(model.output, model.state_0)
                    
                    cnt += LQ.size(0)#val_data["LQ"].size(0)
                    
                    # visualize first image in the batch
                    visuals = model.get_current_visuals(0)
#                     print('visuals["Output"] shape',visuals["Output"].shape)
                    output = visuals["Output"].squeeze()#util.tensor2img(visuals["Output"].squeeze())  # uint8
#                     print('output shape:', output.shape)
                    # save the validation results
                    if _ % 200 == 0.0:
                        logger.info("Validation pics saving ")#.format(_)
                # debug print
                        print('min max values of model output:',torch.min(output),torch.max(output))
                        print('min max values of LR:',torch.min(LQ),torch.max(LQ))
                        print('min max values of HR:',torch.min(GT),torch.max(GT))
                        LQ = denorm(LQ)
                        GT = denorm(GT)
                        output = denorm(output)
                        # add debug print
                        print('min max values of LR after denormalization:',torch.min(LQ),torch.max(LQ))
                        print('min max values of HR after denormalization:',torch.min(GT),torch.max(GT))
                        print('min max values of model output after denormalization:',torch.min(output),torch.max(output))
                        
                        if not opt['datasets']["train"]['only_IR']:
                            visualize_save_triplet_rgb(LQ[0], GT[0], output, idx = _, cur_state = str(current_step), fuse =opt['datasets']["train"]['fuse'],  path = opt['datasets']["val"]['save_dir'])
                        if opt['datasets']["train"]['fuse']:
                            visualize_save_triplet_ir(LQ[0], GT[0], output, idx = _, cur_state = str(current_step), path = opt['datasets']["val"]['save_dir'])

                    # calculate PSNR
#                     gt_img = GT[0,4:7,:,:]# only rgb
#                     print('output shape', output.shape)
#                     print('gt_img shape', gt_img.shape)
#                     psnr = util.calculate_psnr(output[4:7,:,:], gt_img)
#                     ssim = util.calculate_ssim(output[4:7,:,:].permute(1,2,0), gt_img.permute(1,2,0))
#                     print('psnr', psnr)
#                     print('ssim', ssim)
#                     avg_psnr += util.calculate_psnr(output[4:7,:,:], gt_img)
# #                     print('psnr evaled')
#                     avg_ssim += util.calculate_ssim(output[4:7,:,:].permute(1,2,0), gt_img.permute(1,2,0))
# #                     print('ssim evaled')
                    idx += 1
                        
                if opt["dist"]:
                    dist.all_reduce(cnt)
                    dist.all_reduce(avg_psnr)
                    dist.all_reduce(avg_ssim)
                
                if rank <= 0:    
                    avg_psnr = avg_psnr / cnt
                    avg_ssim = avg_ssim / cnt
                    
                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        best_iter = current_step
                    
                    if avg_ssim > best_ssim:
                        best_ssim = avg_ssim
                        best_ssim_iter = current_step

                    # log
                    logger.info("# Validation \n" +
                                "        # PSNR: {:.4f}, best {:.4f} @ iter {}\n"
                                "        # SSIM: {:.4f}, best {:.4f} @ iter {}"
                                .format(avg_psnr, best_psnr, best_iter, 
                                        avg_ssim, best_ssim, best_ssim_iter))
                    logger_val = logging.getLogger("val")  # validation logger
                    logger_val.info(
                        "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, ssim: {:.4f}>".format(
                            epoch, current_step, avg_psnr, avg_ssim
                        )
                    )
                    print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}, ssim: {:.4f}>".format(
                            epoch, current_step, avg_psnr, avg_ssim
                        ))
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        tb_logger.add_scalar("psnr", avg_psnr, current_step)
                        tb_logger.add_scalar("ssim", avg_ssim, current_step)
                    print("psnr", avg_psnr)
                    print("ssim", avg_ssim)

            if error.value:
                sys.exit(0)
#             #### save models and training states
#             if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
#                 if rank <= 0:
#                     logger.info("Saving models and training states.")
#                     model.save(current_step)
#                     model.save_training_state(epoch, current_step)
# #                 logger.info('model saved after the epoch=',epoch)
#                     print('model saved after the epoch=',epoch)
#         print('Saving model saved on epoch=',epoch)
#         model.save(current_step)
#         model.save_training_state(epoch, current_step)
    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
