import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
import vtk
from vtk_utils.vtk_utils import *
import dataset
from torch.utils.data import DataLoader
import yaml
import functools
from gen_network import SDF4CHD, SDF4CHDTester, LipLinearLayer
import pkbar
import matplotlib.pyplot as plt
from io_utils import plot_loss_curves, save_ckp, write_sampled_point, load_ckp
import io_utils
import argparse
import h5py
import random
import math
import net_utils
from torchinfo import summary
from network import act
from dataset import sample_points_from_sdf
from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print("DEVICE: ", device)

def loss_func(weights, sampled_gt_sdv, sampled_gt_distance, chd_type, type_z, type_s, outputs, kbar, i):
    recons_noDs_loss = torch.mean(((outputs['recons'][0].squeeze(1) - sampled_gt_sdv)**2))
    recons_loss = torch.mean(((outputs['recons'][1].squeeze(1) - sampled_gt_sdv)**2))
    gaussian_t_loss = torch.mean(type_z**2)
    gaussian_s_loss = torch.mean(type_s**2)
    if weights['div_integral'] > 0.:
        div_integral = torch.mean(outputs['div_integral'])
    else:
        div_integral = torch.tensor(0., dtype=torch.float32).to(device)
    
    if weights['grad_mag'] > 0.:
        grad_mag = torch.mean(outputs['grad_mag'] * torch.clamp(torch.min(sampled_gt_distance, dim=1)[0], min=0.)) 
    else:
        grad_mag = torch.tensor(0., dtype=torch.float32).to(device)

    total_loss = weights['recons_loss'] * (recons_loss) + \
            weights['recons_noDs_loss'] * recons_noDs_loss + \
            weights['gaussian_t_loss'] * gaussian_t_loss + \
            weights['gaussian_s_loss'] * gaussian_s_loss + \
            weights['div_integral'] * div_integral + \
            weights['grad_mag'] * grad_mag

    kbar.update(i, values=[("loss", total_loss), ("recons", recons_loss), ("recons_noDs", recons_noDs_loss),  
        ("gaussian_s_loss", gaussian_s_loss), ("gaussian_t_loss", gaussian_t_loss), ("div_integral", div_integral), ("grad_mag", grad_mag)])
    writer.add_scalar("Loss/total_loss", total_loss)
    writer.add_scalar("Loss/recons_loss", recons_loss)
    writer.add_scalar("Loss/recons_noDs_loss", recons_noDs_loss)
    writer.add_scalar("Loss/div_integral", div_integral)
    return total_loss, recons_noDs_loss

def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def initialize_type_network(cfg, net, optimizer_nodecay):
    # initilize to fit a heart regardless of the type first
    sdf_py_tmplt = pickle.load(open(cfg['data']['tmplt_sdf'], 'rb'))
    for i in range(500):
        _, points, point_values, _ = sample_points_from_sdf(sdf_py_tmplt, cfg['train']['n_smpl_pts'], cfg['data']['point_sampling_factor'])
        points = points.unsqueeze(0).to(device)
        point_values = point_values.unsqueeze(0).to(device)
        chd_type = (torch.rand((1, len(cfg['data']['chd_info']['types'])))>0.5).float().to(device)
        if net.module.use_diag:
            z_t = chd_type
        else:
            z_t = net.module.type_encoder(chd_type)
        out = act(net.module.decoder.decoder(z_t, points))
        recons_loss = torch.mean(((out.permute(0, 2, 1) - point_values)**2)*(point_values+1))
        recons_loss.backward()
        print("ITER {}: Recons loss: {}.".format(i, recons_loss.item()))
        optimizer_nodecay.step()
    return net

def regularize_lip_bound(net):
    prod_c_0 = 1.
    for layer in net.module.decoder.decoder.children():
        if isinstance(layer, LipLinearLayer):
            prod_c_0 = prod_c_0 * F.softplus(layer.c)
    return prod_c_0

def update_prediction(dataloader, lat_vecs, lat_vecs_ds, net, cfg, epoch):
    tester = SDF4CHDTester(device, cell_grid_size=1, out_dim=cfg['net']['out_dim'])
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if data['filename'][0] in ['ct_1001_image', 'ct_1007_image', 'ct_1010_image', 'ct_1015_image', 'ct_1025_image', 'ct_1042_image', 'ct_1052_image']:
                chd_type = data['chd_type'].to(device)
                if net.module.use_diag:
                    z_t = chd_type
                else:
                    z_t = net.module.type_encoder(chd_type)
                z_s = lat_vecs(data['idx'].to(device))
                z_s = z_s.view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
                z_s_ds = lat_vecs_ds(data['idx'].to(device))
                z_s_ds = z_s_ds.view(1, cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
                sdf_noCorr = tester.z2voxel(z_s, z_s_ds, z_t, net.module, num_blocks=1, out_block=0, get_tmplt_coords=False)
                sdf_Corr = tester.z2voxel(z_s, z_s_ds, z_t, net.module, num_blocks=1, out_block=1, get_tmplt_coords=False)
                type_sdf = tester.z2voxel(None, None, z_t, net.module, num_blocks=1, out_type=True)
                io_utils.write_sdf_to_vtk_mesh(sdf_noCorr, os.path.join(cfg['data']['output_dir'], 'noCorr_{}_epoch{}.vtp'.format(data['filename'][0], epoch)), 0.5) 
                io_utils.write_sdf_to_vtk_mesh(sdf_Corr, os.path.join(cfg['data']['output_dir'], 'Corr_{}_epoch{}.vtp'.format(data['filename'][0], epoch)), 0.5) 
                io_utils.write_sdf_to_vtk_mesh(type_sdf, os.path.join(cfg['data']['output_dir'], 'type_{}_epoch{}.vtp'.format(data['filename'][0], epoch)), 0.5) 
    net.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config') 
    args = parser.parse_args()
    start_epoch = 0
    mode = ['train']
    use_aug = False
    use_error = False
    wt_dcy = 0.
    with open(args.config, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)

    if not os.path.exists(cfg['data']['output_dir']):
        os.makedirs(cfg['data']['output_dir'])
    
    writer = SummaryWriter(cfg['data']['output_dir'])
    two_shape_codes = cfg['net']['two_shape_codes']

    # create dataloader
    train = dataset.SDFDataset(cfg['data']['train_dir'], cfg['train']['n_smpl_pts'], cfg['data']['output_dir'], cfg['data']['point_sampling_factor'], \
        cfg['data']['chd_info'], mode=mode, use_aug=use_aug, use_error=use_error, train=True, pad_num=cfg['train']['pad_num'], \
        binary=cfg['train']['binary'])
    dataloader_train = DataLoader(train, batch_size=cfg['train']['batch_size'], shuffle=True, pin_memory=True, drop_last=False, worker_init_fn = worker_init_fn, num_workers=min(cfg['train']['batch_size'], 4))
    dataloader_val = DataLoader(train, batch_size=1, shuffle=False, pin_memory=True, drop_last=True, worker_init_fn = worker_init_fn, num_workers=0)

    # create network and latent codes
    net = SDF4CHD(in_dim=0, \
            out_dim=cfg['net']['out_dim'], \
            num_types=len(cfg['data']['chd_info']['types']), \
            z_t_dim=cfg['net']['z_t_dim'], \
            z_s_dim=cfg['net']['z_s_dim'], \
            type_mlp_num=cfg['net']['type_mlp_num'],\
            ds_mlp_num=cfg['net']['ds_mlp_num'],\
            dx_mlp_num=cfg['net']['dx_mlp_num'], \
            latent_dim=cfg['net']['latent_dim'], \
            ins_norm=cfg['net']['ins_norm'], \
            type_bias=False, \
            lip_reg=cfg['net']['lip_reg'], \
            step_size=cfg['net']['step_size'], \
            use_diag=cfg['net']['use_diag'], \
            div_loss=True if cfg['train']['weights']['div_integral']> 0. else False, \
            act_func=net_utils.act if cfg['train']['binary'] else lambda x: x)
    # initialize Z_s
    lat_vecs = torch.nn.Embedding(len(train.idx_dict), cfg['net']['z_s_dim']*cfg['net']['l_dim']*cfg['net']['l_dim']*cfg['net']['l_dim'], max_norm=1.).to(device)
    torch.nn.init.kaiming_normal_(lat_vecs.weight.data, a=0.02, nonlinearity='leaky_relu')
    if two_shape_codes:
        lat_vecs_ds = torch.nn.Embedding(len(train.idx_dict), cfg['net']['z_s_dim']*cfg['net']['l_dim']*cfg['net']['l_dim']*cfg['net']['l_dim'], max_norm=1.).to(device)
        torch.nn.init.kaiming_normal_(lat_vecs_ds.weight.data, a=0.02, nonlinearity='leaky_relu')
        zs_params = list(lat_vecs.parameters())+list(lat_vecs_ds.parameters())
    else:
        zs_params = list(lat_vecs.parameters())
    net = nn.DataParallel(net)
    net.to(device)
   
    # no weight decay for type prediction
    subnets_type = ['decoder.decoder', 'type_encoder']
    params_type, params_else = set(), set()
    for n in subnets_type:
        params_type |= set(nm for nm, p in net.named_parameters() if n in nm)
    all_names = set(nm for nm, p in net.named_parameters())
    params_else = all_names.difference(params_type)
    
    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = params_type & params_else
    union_params = params_type | params_else 
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    
    optimizer_nodecay = torch.optim.Adam((params_dict[n] for n in sorted(list(params_type))), lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_decay = torch.optim.Adam((params_dict[n] for n in sorted(list(params_else))), lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=wt_dcy)
    optimizer_zs = torch.optim.Adam(zs_params, lr=cfg['train']['lr'], betas=(0.5, 0.999), weight_decay=0.0)  

    scheduler_nodecay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nodecay, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    scheduler_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decay, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    scheduler_zs = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_zs, patience=cfg['train']['scheduler']['patience'], factor=cfg['train']['scheduler']['factor'], verbose=True, min_lr=1e-6)
    optimizers = [optimizer_nodecay, optimizer_decay, optimizer_zs]
    schedulers = [scheduler_nodecay, scheduler_decay, scheduler_zs]
    
    if os.path.exists(os.path.join(cfg['data']['output_dir'], 'net.pt')):
        print("LOADING LASTEST CHECKPOINT")
        net, optimizers, schedulers, start_epoch = io_utils.load_ckp(os.path.join(cfg['data']['output_dir'], 'net.pt'), \
                net, optimizers, schedulers)
        lat_vecs.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code.pt'))['latent_codes'])
        if two_shape_codes:
            lat_vecs_ds.load_state_dict(torch.load(os.path.join(cfg['data']['output_dir'], 'code_ds.pt'))['latent_codes'])
    else:
        ## initialize type network - helps with convergence
        if cfg['train']['init']:
            for n in params_else:
                params_dict[n].requires_grad = False
                params_dict[n].grad = None
            net = initialize_type_network(cfg, net, optimizer_nodecay)
            for n in params_else:
                params_dict[n].requires_grad = True
        else:
            pass
    
    fix_type = False
    # start training
    for epoch in range(start_epoch, cfg['train']['epoch']):
        kbar = pkbar.Kbar(target=len(dataloader_train), epoch=epoch, num_epochs=cfg['train']['epoch'], width=20, always_stateful=False)
        net.train()
        if cfg['train']['alter']:
            if epoch >= cfg['train']['alter']['joint_num'] and epoch < cfg['train']['alter']['end_num']:
                if epoch % cfg['train']['alter']['alter_num'] == 0:
                    print("RESETING Optimizer")
                    optimizers[0] = torch.optim.Adam((params_dict[n] for n in sorted(list(params_type))), lr=optimizers[0].param_groups[0]['lr'], betas=(0.5, 0.999), weight_decay=0.0)
                    optimizers[1] = torch.optim.Adam((params_dict[n] for n in sorted(list(params_else))), lr=optimizers[1].param_groups[0]['lr'], betas=(0.5, 0.999), weight_decay=wt_dcy)
                    optimizers[2] = torch.optim.Adam(zs_params, lr=optimizers[2].param_groups[0]['lr'], betas=(0.5, 0.999), weight_decay=0.0)
                if (((epoch-cfg['train']['alter']['joint_num']) // cfg['train']['alter']['alter_num']) % 2 == 1):
                    print("TRAIN TYPE ONLY")
                    fix_type = False
                    for n in params_type:
                        params_dict[n].requires_grad = True
                    for n in params_else:
                        params_dict[n].requires_grad = False
                        params_dict[n].grad = None
                    for p in zs_params:
                        p.requires_grad = False
                        p.grad = None
                else:
                    fix_type = True
                    print("TRAIN SHAPE ONLY, Updating TYPE BOUNDARY POINTS")
                    for n in params_else:
                        params_dict[n].requires_grad = True
                    for n in params_type:
                        params_dict[n].requires_grad = False
                        params_dict[n].grad = None
                    for p in zs_params:
                        p.requires_grad = True
            elif epoch >= cfg['train']['alter']['end_num']:
                print("TRAIN TYPE AND SHAPE")
                fix_type = False
                for n in params_else:
                    params_dict[n].requires_grad = True
                for n in params_type:
                    params_dict[n].requires_grad = True
                for p in zs_params:
                    p.requires_grad = True
        total_recons_noDs_loss = 0.
        for i, data in enumerate(dataloader_train):
            z_s = lat_vecs(data['idx'].to(device))
            z_s = z_s.view(data['points'].shape[0], cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
            if two_shape_codes:
                z_s_ds = lat_vecs_ds(data['idx'].to(device))
                z_s_ds = z_s_ds.view(data['points'].shape[0], cfg['net']['z_s_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'], cfg['net']['l_dim'])
            points = data['points'].to(device)
            point_values = data['pt_sdv'].to(device)
            point_values_sdv = data['pt_distance'].to(device)
            chd_type = data['chd_type'].to(device)
            if epoch == start_epoch and i ==0:
                summary(net, [tuple(z_s.shape), tuple(z_s.shape), tuple(points.shape), tuple(chd_type.shape)])
                print(points.shape, point_values.shape)
                io_utils.write_sampled_point(points[0], point_values[0], os.path.join(cfg['data']['output_dir'], 'sample_{}_epoch{}.vtp'.format(i, epoch)))
            net.zero_grad()
            if two_shape_codes:
                outputs, z_t = net(z_s, z_s_ds, points, chd_type) 
                loss, recons_noDs_loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, z_t, torch.cat([z_s, z_s_ds], dim=-1), outputs, kbar, i)
            else:
                outputs, z_t = net(z_s, z_s, points, chd_type)
                loss, recons_noDs_loss = loss_func(cfg['train']['weights'], point_values, point_values_sdv, chd_type, z_t, z_s, outputs, kbar, i)
            total_recons_noDs_loss += recons_noDs_loss.item()
            if cfg['net']['lip_reg']:
                loss = loss + cfg['train']['lip_weight']*regularize_lip_bound(net)
            loss.backward()
            for o in optimizers:
                o.step()
        with torch.no_grad():
            for s in schedulers:
                s.step(total_recons_noDs_loss)
            if (epoch+1) % cfg['train']['save_every'] == 0:
                all_latents = lat_vecs.state_dict()
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code{}.pt'.format(epoch+1)))
                torch.save({'epoch': epoch+1, 'latent_codes': all_latents}, os.path.join(cfg['data']['output_dir'], 'code.pt'))
                if two_shape_codes:
                    all_latents_ds = lat_vecs_ds.state_dict()
                    torch.save({'epoch': epoch+1, 'latent_codes': all_latents_ds}, os.path.join(cfg['data']['output_dir'], 'code_ds{}.pt'.format(epoch+1)))
                    torch.save({'epoch': epoch+1, 'latent_codes': all_latents_ds}, os.path.join(cfg['data']['output_dir'], 'code_ds.pt'))
                save_ckp(net, optimizers, schedulers, epoch, cfg['data']['output_dir']) 
                if two_shape_codes:
                    update_prediction(dataloader_val, lat_vecs, lat_vecs_ds, net, cfg, epoch)
                else:
                    update_prediction(dataloader_val, lat_vecs, lat_vecs, net, cfg, epoch)
    writer.flush()
    writer.close()
