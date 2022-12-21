import time
import shutil
from nst_net import NST_Net
from rendering import *
from tqdm import tqdm
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.autograd import Variable 
from dataset import RaySampler, StyleRaySampler_gen, LightDataLoader
from learnable_latents import VAE, Learnable_Latents
from style_nerf import Style_NeRF, Style_Module
from config import config_parser
from utils import mse2psnr, img2mse, batchify
from sample import sampling_pts_uniform, sampling_pts_fine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pretrain_nerf(args, global_step, samp_func, samp_func_fine, nerf, nerf_fine, nerf_optimizer, ckpt_dir_nerf, sv_path):

    train_dataset = RaySampler(data_path=args.datadir, factor=args.factor,
                                   mode='train', valid_factor=args.valid_factor,no_ndc=args.no_ndc,
                                   pixel_alignment=args.pixel_alignment, spherify=args.spherify)
    print('Finish create dataset')
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=(args.num_workers > 0), generator=torch.Generator(device=device))

    """batchify nerf"""
    nerf_forward = batchify(lambda **kwargs: nerf(**kwargs), args.chunk)
    nerf_forward_fine = batchify(lambda **kwargs: nerf_fine(**kwargs), args.chunk)

    """Render valid for nerf"""
    if args.render_valid:
        render_path = sv_path / ('render_valid_' + str(global_step))
        valid_dataset = train_dataset
        valid_dataset.set_mode('valid')
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
        with torch.no_grad():
            if args.N_samples_fine > 0:
                rgb_map, t_map, _, _ = render(nerf_forward=nerf_forward, samp_func=samp_func, dataloader=valid_dataloader,
                                              args=args, device=device, sv_path=render_path, nerf_forward_fine=nerf_forward_fine,
                                              samp_func_fine=samp_func_fine)
            else:
                rgb_map, t_map, _, _ = render(nerf_forward=nerf_forward, samp_func=samp_func, dataloader=valid_dataloader,
                                              args=args, device=device, sv_path=render_path)
        print('Done, saving', rgb_map.shape, t_map.shape)
        exit(0)
    
    """Render train for nerf"""
    if args.render_train:
        render_path =sv_path / ('render_train_' + str(global_step))
        render_dataset = train_dataset
        if args.N_samples_fine > 0:
            render_train(samp_func=samp_func, nerf_forward=nerf_forward, dataset=render_dataset, args=args, device=device, sv_path=render_path, nerf_forward_fine=nerf_forward_fine, samp_func_fine=samp_func_fine)
        else:
            render_train(samp_func=samp_func, nerf_forward=nerf_forward, dataset=render_dataset, args=args, device=device, sv_path=render_path)
        exit(0)

    """Training Loop"""
    print('start training')
    # Elapse Measurement
    data_time, model_time, opt_time = 0, 0, 0
    fine_time = 0
    while True:
        for batch_data in tqdm(train_dataloader):
            # Get batch data
            start_t = time.time()
            rgb_gt, rays_o, rays_d = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d']

            pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=train_dataset.near, far=train_dataset.far, perturb=True)
            ray_num, pts_num = rays_o.shape[0], args.N_samples
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
            # Forward and Composition
            forward_t = time.time()
            ret = nerf_forward(pts=pts, dirs=rays_d_forward)
            pts_rgb, pts_sigma = ret['rgb'], ret['sigma']
            rgb_exp, t_exp, weights = alpha_composition(pts_rgb, pts_sigma, ts, args.sigma_noise_std)

            # Calculate Loss
            loss_rgb = img2mse(rgb_gt, rgb_exp)
            loss = loss_rgb

            fine_t = time.time()
            if args.N_samples_fine > 0:
                pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
                pts_num = args.N_samples + args.N_samples_fine
                rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
                ret = nerf_forward_fine(pts=pts_fine, dirs=rays_d_forward)
                pts_rgb_fine, pts_sigma_fine = ret['rgb'], ret['sigma']
                rgb_exp_fine, t_exp_fine, _ = alpha_composition(pts_rgb_fine, pts_sigma_fine, ts_fine, args.sigma_noise_std)
                loss_rgb_fine = img2mse(rgb_gt, rgb_exp_fine)
                loss = loss + loss_rgb_fine

            # Backward and Optimize
            nerf_optimizer.zero_grad()
            loss.backward()
            nerf_optimizer.step()

            if global_step % args.i_print == 0:
                psnr = mse2psnr(loss_rgb)
                if args.N_samples_fine > 0:
                    psnr_fine = mse2psnr(loss_rgb_fine)
                    tqdm.write(
                        f"[ORIGIN TRAIN] Iter: {global_step} Loss: {loss.item()} PSNR: {psnr.item()} PSNR Fine: {psnr_fine.item()} RGB Loss: {loss_rgb.item()} RGB Fine Loss: {loss_rgb_fine.item()}"
                        f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")
                else:
                    tqdm.write(
                        f"[ORIGIN TRAIN] Iter: {global_step} Loss: {loss_rgb.item()} PSNR: {psnr.item()} RGB Loss: {loss_rgb.item()}"
                        f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")

                data_time, model_time, opt_time = 0, 0, 0
                fine_time = 0

            # Update Learning Rate
            decay_rate = 0.1
            decay_steps = args.lr_decay
            new_lr = args.lr * (decay_rate ** (global_step / decay_steps))
            for param_group in nerf_optimizer.param_groups:
                param_group['lr'] = new_lr

            # Time Measuring
            end_t = time.time()
            data_time += (forward_t - start_t)
            model_time += (fine_t - forward_t)
            fine_time += 0
            opt_time += (end_t - fine_t)

            # Rest is logging
            if global_step % args.i_weights == 0 and global_step > 0 or global_step >= args.origin_step:
                path = ckpt_dir_nerf / '{:06d}.tar'.format(global_step)
                if args.N_samples_fine > 0:
                    torch.save({
                        'global_step': global_step,
                        'nerf': nerf.state_dict(),
                        'nerf_fine': nerf_fine.state_dict(),
                        'nerf_optimizer': nerf_optimizer.state_dict()
                    }, path)
                else:
                    torch.save({
                        'global_step': global_step,
                        'nerf': nerf.state_dict(),
                        'nerf_optimizer': nerf_optimizer.state_dict(),
                    }, path)
                print('Saved checkpoints at', path)

                # Delete ckpts
                ckpts = [ f for f in sorted(ckpt_dir_nerf.glob('*')) if 'tar' in str(f)]
                if len(ckpts) > args.ckp_num:
                    os.remove(ckpts[0])

            global_step += 1
            if global_step > args.origin_step:
                return global_step
   
def gen_nerf_images(args, samp_func, samp_func_fine, nerf, nerf_fine, nerf_gen_data_path):
    """set nerf"""
    nerf_forward = batchify(lambda **kwargs: nerf(**kwargs), args.chunk)
    if args.N_samples_fine > 0:
        nerf_forward_fine = batchify(lambda **kwargs: nerf_fine(**kwargs), args.chunk)

    """Dataset Creation"""
    tmp_dataset = RaySampler(data_path=args.datadir, factor=args.factor,
                                  mode='valid', valid_factor=args.gen_factor,
                                     no_ndc=args.no_ndc, pixel_alignment=args.pixel_alignment, spherify=args.spherify)

                                     
    tmp_dataloader = DataLoader(tmp_dataset, args.batch_size_style, shuffle=False, num_workers=args.num_workers,
                                pin_memory=(args.num_workers > 0))
    print("Preparing nerf data for style training ...")
    render_nerf_for_nst(nerf_forward=nerf_forward, samp_func=samp_func, dataloader=tmp_dataloader, args=args,
                 sv_path=nerf_gen_data_path, nerf_forward_fine=nerf_forward_fine, samp_func_fine=samp_func_fine)

    return


def train_style_nerf(args, global_step, samp_func, samp_func_fine, nerf, nerf_fine, sv_path, nerf_gen_data_path):
    """batchify nerf"""
    nerf_forward = batchify(lambda **kwargs: nerf(**kwargs), args.chunk)
    nerf_forward_fine = batchify(lambda **kwargs: nerf_fine(**kwargs), args.chunk)

    """Create Style Module"""
    style_model = Style_Module(args)
    style_model.train()
    style_vars = style_model.parameters()
    style_forward = batchify(lambda **kwargs: style_model(**kwargs), args.chunk)
    style_optimizer = torch.optim.Adam(params=style_vars, lr=args.lr, betas=(0.9, 0.999))

    
    """Load Check Point for style module"""
    ckpt_dir_style = sv_path / 'style'
    save_makedir(ckpt_dir_style)
    ckpts_style = [f for f in sorted(ckpt_dir_style.glob('*')) if 'tar' in str(f) and 'style' in str(f) and 'latent' not in str(f)]
    if len(ckpts_style) > 0 and not args.no_reload:
        ckpt_path_style = ckpts_style[-1]
        print('Reloading Style Model from ', ckpt_path_style)
        ckpt_style = torch.load(ckpt_path_style)
        global_step = ckpt_style['global_step']
        style_model.load_state_dict(ckpt_style['model'])
        style_optimizer.load_state_dict(ckpt_style['optimizer'])
    

    """Dataset Creation"""
    train_dataset = StyleRaySampler_gen(data_path=args.datadir, gen_path=nerf_gen_data_path, style_path=args.styledir,
                                        factor=args.factor,
                                        mode='train', valid_factor=args.valid_factor, 
                                        no_ndc=args.no_ndc,
                                        pixel_alignment=args.pixel_alignment, spherify=args.spherify,
                                        decoder_dir=args.ckpt_dir_decoder, no_reload=args.no_reload)
    train_dataset.collect_all_stylized_images()
    train_dataset.set_mode('train_style')

    """Dataloader Preparation"""   
    train_dataloader = LightDataLoader(train_dataset, batch_size=args.batch_size_style, shuffle=True, \
                                        num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
    rounds_per_epoch = int(train_dataloader.data_num / train_dataloader.batch_size)
    print('DataLoader Creation Done !')
                                  
    """VAE"""
    vae = VAE(data_dim=1024, latent_dim=args.vae_latent, W=args.vae_w, D=args.vae_d,
              kl_lambda=args.vae_kl_lambda)
    vae.eval()
    vae_ckpt = args.vae_pth_path
    vae.load_state_dict(torch.load(vae_ckpt)['vae'])

    """Latents Module"""
    latents_model = Learnable_Latents(style_num=train_dataset.style_num, frame_num=train_dataset.frame_num, latent_dim=args.vae_latent)
    vae.to(device)
    ckpt_dir_latent = sv_path / 'latent'
    latent_ckpts = [ f for f in sorted(ckpt_dir_latent.glob('*')) if 'tar' in str(f) and 'style' not in str(f) and 'latent' in str(f)]
    print('Found ckpts', latent_ckpts, ' from ', ckpt_dir_latent, ' For Latents Module.')
    if len(latent_ckpts) > 0 and not args.no_reload:
        latent_ckpt_path = latent_ckpts[-1]
        print('Reloading Latent Model from ', latent_ckpt_path)
        latent_ckpt = torch.load(latent_ckpt_path)
        latents_model.load_state_dict(latent_ckpt['train_set'])
    else:
        vae.to(device)
        print("Initializing Latent Model")
        # Calculate and Initialize Style Latents
        all_style_features = torch.from_numpy(train_dataset.style_features).float().to(device)
        _, style_latents_mu, style_latents_sigma = vae.encode(all_style_features)
        # Set Latents
        style_latents_mu = style_latents_mu.detach()
        style_latents_sigma = style_latents_sigma.detach()
        latents_model.set_latents(style_latents_mu, style_latents_sigma)

    # Render valid style
    if args.render_valid_style:
        render_path = sv_path / ('render_valid_' + str(global_step))
        # Enable style
        nerf.set_enable_style(True)
        if args.N_samples_fine > 0:
            nerf_fine.set_enable_style(True)
        valid_dataset = train_dataset
        valid_dataset.mode = 'valid_style'
        valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(args.num_workers > 0))
        with torch.no_grad():
            if args.N_samples_fine > 0:
                rgb_map, t_map, _, _ = render_style(nerf_forward=nerf_forward, samp_func=samp_func, style_forward=style_forward, latents_model=latents_model,
                                                                        dataloader=valid_dataloader, args=args, device=device, sv_path=render_path,
                                                                        nerf_forward_fine=nerf_forward_fine, samp_func_fine=samp_func_fine, sigma_scale=args.sigma_scale)
            else:
                rgb_map, t_map, _, _ = render_style(nerf_forward=nerf_forward, samp_func=samp_func, style_forward=style_forward, latents_model=latents_model, dataloader=valid_dataloader,
                                                    args=args, device=device, sv_path=render_path, sigma_scale=args.sigma_scale)
        print('Done, saving', rgb_map.shape, t_map.shape)
        return

    # Render train style
    if args.render_train_style:
        render_path = sv_path / ('render_train_' + str(global_step))
        # Enable style
        nerf.set_enable_style(True)
        if args.N_samples_fine > 0:
            nerf_fine.set_enable_style(True)
        render_dataset = train_dataset
        render_dataset.mode = 'train_style'
        if args.N_samples_fine > 0:
            render_train_style(samp_func=samp_func, nerf_forward=nerf_forward, style_forward=style_forward, latents_model=latents_model, dataset=render_dataset, args=args, device=device, sv_path=render_path, nerf_forward_fine=nerf_forward_fine, samp_func_fine=samp_func_fine, sigma_scale=args.sigma_scale)
        else:
            render_train_style(samp_func=samp_func, nerf_forward=nerf_forward, style_forward=style_forward, latents_model=latents_model, dataset=render_dataset, args=args, device=device, sv_path=render_path, sigma_scale=args.sigma_scale)
        return

    # Elapse Measurement
    data_time, model_time, opt_time = 0, 0, 0
    fine_time = 0

    """NST Net"""
    nst_net = NST_Net(args.vgg_pth_path)
    ckpt_dir_decoder = Path(args.ckpt_dir_decoder)
    ckpts = [ f for f in sorted(ckpt_dir_decoder.glob('*')) if 'decoder_iter_' in str(f)]
    if len(ckpts) > 0 and not args.no_reload:
        print(f'loading {ckpts[-1]}')
        ld_dict = torch.load(ckpts[-1])
        nst_net.load_decoder_state_dict(ld_dict['decoder'])
    else:
        print('Please finetune decoder first')
        exit(0)
    nst_net.eval()
    nst_net.to(device)
    

    """Model Mode for Style"""
    nerf_fine.eval()
    nerf.eval()

    latents_model.set_latents_optim()

    while True:
        for _ in range(rounds_per_epoch):
            batch_data = train_dataloader.get_batch()

            # Get batch data
            start_t = time.time()
            rgb_gt, rays_o, rays_d, rgb_origin = batch_data['rgb_gt'], batch_data['rays_o'], batch_data['rays_d'], batch_data['rgb_origin']
            style_id, frame_id = batch_data['style_id'].long(), batch_data['frame_id'].long()

            # Sample
            pts, ts = samp_func(rays_o=rays_o, rays_d=rays_d, N_samples=args.N_samples, near=train_dataset.near, far=train_dataset.far, perturb=True)
            ray_num, pts_num = rays_o.shape[0], args.N_samples
            rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])

            # Forward
            forward_t = time.time()
            ret = nerf_forward(pts=pts, dirs=rays_d_forward)
            pts_sigma, pts_embed = ret['sigma'], ret['pts']
            # Stylize
            style_latents = latents_model(style_ids=style_id, frame_ids=frame_id)
            style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
            ret_style = style_forward(x=pts_embed, latent=style_latents_forward)
            pts_rgb_style = ret_style['rgb']
            # Composition
            rgb_exp_style, _, weights = alpha_composition(pts_rgb_style, pts_sigma, ts, args.sigma_noise_std)
            # Pixel-wise Loss
            loss_rgb = args.rgb_loss_lambda * img2mse(rgb_exp_style, rgb_gt)
            # Latent LogP loss
            logp_loss_lambda = args.logp_loss_lambda * (args.logp_loss_decay ** int((global_step - args.origin_step) / 1000))
            loss_logp = logp_loss_lambda * latents_model.loss(style_ids=style_id, frame_ids=frame_id)

            fine_t = time.time()
            if args.N_samples_fine > 0:
                # Sample
                pts_fine, ts_fine = samp_func_fine(rays_o, rays_d, ts, weights, args.N_samples_fine)
                pts_num = args.N_samples + args.N_samples_fine
                rays_d_forward = rays_d.unsqueeze(1).expand([ray_num, pts_num, 3])
                # Forward
                ret = nerf_forward_fine(pts=pts_fine, dirs=rays_d_forward)
                pts_sigma_fine, pts_embed_fine = ret['sigma'], ret['pts']
                # Stylize
                style_latents_forward = style_latents.unsqueeze(1).expand([ray_num, pts_num, style_latents.shape[-1]])
                ret_style = style_forward(x=pts_embed_fine, latent=style_latents_forward)
                pts_rgb_style_fine = ret_style['rgb']
                # Composition
                rgb_exp_style_fine, _, _ = alpha_composition(pts_rgb_style_fine, pts_sigma_fine, ts_fine, args.sigma_noise_std)
                loss_rgb_fine = args.rgb_loss_lambda * img2mse(rgb_exp_style_fine, rgb_gt)
                loss_rgb += loss_rgb_fine

            # Loss for stylized NeRF
            loss_mimic = loss_rgb
            loss = loss_mimic + loss_logp

            # Backward and Optimize
            opt_t = time.time()
            style_optimizer.zero_grad()
            loss.backward()
            style_optimizer.step()
            latents_model.optimize()


            # Time Measuring
            end_t = time.time()
            data_time += (forward_t - start_t)
            model_time += (fine_t - forward_t)
            fine_time += (opt_t - fine_t)
            opt_time += (end_t - fine_t)

            # Rest is logging
            if global_step % args.i_weights == 0 and global_step > 0:
                # Saving Style module
                path = ckpt_dir_style / 'style_{:06d}.tar'.format(global_step)
                torch.save({
                    'global_step': global_step,
                    'model': style_model.state_dict(),
                    'optimizer': style_optimizer.state_dict()
                }, path)
                print('Saved checkpoints at', path)
                # Delete ckpts
                ckpts = [f for f in sorted(ckpt_dir_style.glob('*')) if 'tar' in str(f) and 'style' in str(f) and 'latent' not in str(f)]
                if len(ckpts) > args.ckp_num:
                    os.remove(ckpts[0])

                # Saving Latent Model
                path = ckpt_dir_latent / 'latent_{:06d}.tar'.format(global_step)
                torch.save({
                    'global_step': global_step,
                    'train_set': latents_model.state_dict()
                }, path)
                print('Saved checkpoints at', path)
                # Delete ckpts
                ckpts = [ckpt_dir_latent / f for f in sorted(ckpt_dir_latent) if 'tar' in f and 'style' not in f and 'latent' in f]
                if len(ckpts) > args.ckp_num:
                    os.remove(ckpts[0])


            if global_step % args.i_print == 0:
                tqdm.write(f"[STYLE TRAIN] Iter: {global_step} Loss: {loss.item()} Pixel RGB Loss: {loss_rgb.item()} -Log(p) Loss: {loss_logp.item()}"
                           f" Data time: {np.round(data_time, 2)}s Model time: {np.round(model_time, 2)}s Fine time: {np.round(fine_time, 2)}s Optimization time: {np.round(opt_time, 2)}s")
                data_time, model_time, opt_time = 0, 0, 0
                fine_time = 0

            global_step += 1
            if global_step > args.total_step:
                return global_step

 

def run(args):
    """set sampling functions"""
    samp_func = sampling_pts_uniform
    if args.N_samples_fine > 0:
        samp_func_fine = sampling_pts_fine

    """Saving Configuration"""
    use_viewdir_str = '_UseViewDir_' if args.use_viewdir else ''
    sv_path = Path(args.basedir) / (args.expname + '_' + args.nerf_type + '_' + args.act_type + use_viewdir_str + 'ImgFactor' + str(int(args.factor)))
    print(sv_path)
    save_makedir(sv_path)
    shutil.copy(args.config, sv_path)

    """Create Nerf"""
    nerf = Style_NeRF(args, mode='coarse').to(device)
    nerf.train()
    grad_vars = list(nerf.parameters())
    nerf_fine=None
    if args.N_samples_fine > 0:
        nerf_fine = Style_NeRF(args=args, mode='fine').to(device)
        nerf_fine.train()
        grad_vars += list(nerf_fine.parameters())
    nerf_optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr, betas=(0.9, 0.999))

    """Load Check Point for NeRF"""
    global_step = 0
    ckpt_dir_nerf = sv_path / 'nerf'
    save_makedir(ckpt_dir_nerf)
    ckpts = [f for f in sorted(ckpt_dir_nerf.glob('*')) if 'tar' in str(f) and 'style' not in str(f) and 'latent' not in str(f)]
    print('Found ckpts', ckpts, ' from ', ckpt_dir_nerf)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_nerf = ckpts[-1]
        print('Reloading Nerf Model from ', ckpt_nerf)
        ckpt = torch.load(ckpt_nerf, map_location=device)
        global_step = ckpt['global_step']
        # Load nerf
        nerf.load_state_dict(ckpt['nerf'])
        # Load optimizer
        nerf_optimizer.load_state_dict(ckpt['nerf_optimizer'])
        if args.N_samples_fine > 0:
            nerf_fine.load_state_dict((ckpt['nerf_fine']))

    """For pretrain nerf"""
    if args.pretrain_nerf or args.render_train or args.render_valid:
        global_step = pretrain_nerf(args, global_step=global_step, samp_func=samp_func, samp_func_fine=samp_func_fine, 
                      nerf=nerf, nerf_fine=nerf_fine, nerf_optimizer=nerf_optimizer, ckpt_dir_nerf=ckpt_dir_nerf, sv_path=sv_path)

    nerf_gen_data_path = sv_path / 'nerf_gen_data2/'
    """For generate nerf images"""
    if args.gen_nerf_images:
        gen_nerf_images(args=args, samp_func = samp_func, samp_func_fine=samp_func_fine, nerf=nerf, nerf_fine=nerf_fine,
                         nerf_gen_data_path=nerf_gen_data_path)

    """For train stylenerf"""
    if args.train_style_nerf or args.render_train_style or args.render_valid_style:
        if not nerf_gen_data_path.exists():
            train_decoder_nerf_cmd = './python3 run_stylenerf.py --gen_nerf_images'
            print('{} does not exist, please run {} first.'.format(nerf_gen_data_path, train_decoder_nerf_cmd))
            exit(0)
        global_step = train_style_nerf(args=args, global_step=global_step, samp_func=samp_func, samp_func_fine=samp_func_fine,
                         nerf=nerf, nerf_fine=nerf_fine, sv_path=sv_path, nerf_gen_data_path=nerf_gen_data_path)

    return


if __name__ == '__main__':
    args = config_parser(main_file='run_stylenerf')
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run(args=args)