import configargparse

def config_parser(main_file=None):
    
    if main_file == 'run_stylenerf':
        return config_run_stylenerf()
    elif main_file == 'train_style_modules':
        return config_train_style_module()
    

def config_run_stylenerf():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')

    # data set options
    parser.add_argument("--datadir", type=str, help='input data directory')
    parser.add_argument("--styledir", type=str, default='./all_styles/')
    parser.add_argument("--decoder_pth_path", type=str, default='./pretrained/decoder.pth')
    parser.add_argument("--vgg_pth_path", type=str, default='./pretrained/vgg_normalised.pth')
    parser.add_argument("--vae_pth_path", type=str, default='./pretrained/vae.pth')
    parser.add_argument("--dataset_type", type=str, default='llff')
    parser.add_argument("--factor", type=int, default=1,
                        help='factor to downsample images')
    parser.add_argument("--gen_factor", type=float, default=0.2,  # 5,
                        help='factor for interpolate trace when style training')
    parser.add_argument("--valid_factor", type=float, default=0.05,
                        help='factor for interpolate trace when validating')
    parser.add_argument("--no_ndc", action='store_true', help='No NDC for llff dataset.')
    parser.add_argument("--white_bkgd", action='store_true', help='White Background for blender dataset.')
    parser.add_argument("--half_res", action='store_true', help='Half resolution for linemod dataset.')
    parser.add_argument("--num_workers", type=int, default=0, help='Number of workers for torch dataloader.')
    parser.add_argument("--spherify", action='store_true', help='Spherify camera poses or not')
    parser.add_argument("--store_rays", type=int, default=1,
                        help='factor to downsample images')

    # training options
    parser.add_argument("--use_viewdir", action='store_true',
                        help='use view direction as input.')
    parser.add_argument("--sample_type", type=str, default='uniform',
                        help='Types of sampling: [uniform]')
    parser.add_argument("--act_type", type=str, default='relu',
                        help='Types of activation: [relu, tanh, elu]')
    parser.add_argument("--nerf_type", type=str, default='nerf',
                        help='Types of nerf: [nerf]')
    parser.add_argument("--style_type", type=str, default='mlp',
                        help='Types of style module: [mlp]')
    parser.add_argument("--latent_type", type=str, default='variational',
                        help='Types of latent module: [variational latent]')
    parser.add_argument("--nerf_type_fine", type=str, default='nerf',
                        help='Types of fine nerf: [nerf]')
    parser.add_argument("--sigma_noise_std", type=float, default=1e0,
                        help='std dev of noise added to regularize sigma output, 1e0 recommended')
    parser.add_argument("--siren_sigma_mul", type=float, default=20.,
                        help='amplify positive sigma for siren')

    parser.add_argument("--rgb_loss_lambda", type=float, default=1.,
                        help='Coefficient for style loss')
    parser.add_argument("--rgb_loss_lambda_2d", type=float, default=10.,
                        help='Coefficient for style loss')
    parser.add_argument("--style_loss_lambda", type=float, default=1.,
                        help='Coefficient for style loss')
    parser.add_argument("--content_loss_lambda", type=float, default=1.,
                        help='Coefficient for style loss')
    parser.add_argument("--logp_loss_lambda", type=float, default=0.1,
                        help='Coefficient for logp loss')
    parser.add_argument("--logp_loss_decay", type=float, default=1.,
                        help='Decay rate for logp loss per 1000 steps')
    parser.add_argument("--lambda_u", type=float, default=0.01,
                        help='Nerf in the wild lambda u hyper parameter')

    # Network
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--style_D", type=int, default=8,
                        help='style layers in network')
    parser.add_argument("--style_feature_dim", type=int, default=1024,
                        help='style feature dimension')

    # VAE
    parser.add_argument('--vae_d', type=int, default=4)
    parser.add_argument('--vae_w', type=int, default=512)
    parser.add_argument('--vae_latent', type=int, default=32)
    parser.add_argument('--vae_kl_lambda', type=float, default=0.1)

    parser.add_argument("--embed_freq_coor", type=int, default=10,
                        help='frequency of coordinate embedding')
    parser.add_argument("--embed_freq_dir", type=int, default=4,
                        help='frequency of direction embedding')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--batch_size_style", type=int, default=1024,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lr", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lr_decay", type=int, default=100000,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--total_step", type=int, default=50000001,
                        help='total training step')
    parser.add_argument("--origin_step", type=int, default=250000,
                        help='total training step')
    parser.add_argument("--decoder_step", type=int, default=3500000,
                        help='total training step')
    parser.add_argument("--steps_per_opt", type=int, default=1,
                        help='Steps for gradient accumulation')
    parser.add_argument("--steps_patch", type=int, default=-1,
                        help='Steps interval for patch sampling')

    parser.add_argument("--N_samples", type=int, default=64,
                        help='The number of sampling points per ray')
    parser.add_argument("--N_samples_fine", type=int, default=64,
                        help='The number of sampling points per ray for fine network')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_video",   type=int, default=50000*100,
                        help='frequency of render_poses video saving')
    parser.add_argument("--ckp_num", type=int, default=3,
                        help='Max number of saved ckpts.')

    parser.add_argument("--render_valid",   action='store_true',
                        help='render valid')
    parser.add_argument("--render_train",   action='store_true',
                        help='render train')
    parser.add_argument("--render_valid_style",   action='store_true',
                        help='render valid style')
    parser.add_argument("--render_train_style",   action='store_true',
                        help='render train style')
    parser.add_argument("--sigma_scale", type=float, default=1.)

    # Pixel Alignment
    parser.add_argument("--pixel_alignment", action='store_true',
                        help='Pixel Alignment with half a pixel.')

    parser.add_argument("--TT_far", type=float, default=8., help='Far value of TT dataset NeRF')

    # pretrain nerf
    parser.add_argument("--pretrain_nerf", action="store_true", help="pretrain nerf")
    # train_style_nerf
    parser.add_argument("--train_style_nerf", action="store_true", help="train style nerf")
    # gen nerf images
    parser.add_argument("--gen_nerf_images", action="store_true", help="generate images using origin nerf")

    args = parser.parse_args()
    return args

def config_train_style_module():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vae',
                        help='vae or pretrain_decoder or decoder_with_nerf')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    # Basic options
    parser.add_argument("--datadir", type=str, default='./data/fern', help='input data directory')
    parser.add_argument('--content_dir', type=str, default='./all_contents/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--nerf_content_dir', type=str, default='./log/nerf_gen_data2/',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='./all_styles/',
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg_pretrained_path', type=str, default='./pretrained/vgg_normalised.pth')

    parser.add_argument('--no_ndc', action='store_true')
    parser.add_argument('--no_reload', action='store_true')

    # training options
    parser.add_argument('--save_dir', default='./pretrained/',
                        help='Directory to save the model')
    parser.add_argument('--ckp_num', type=int, default=3)
    parser.add_argument('--log_dir', default='./logs/stylenet/',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000) # origin 160000
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=2.)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--temporal_weight', type=float, default=50.)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=100)
    parser.add_argument('--print_interval', type=int, default=50)
    parser.add_argument("--act_type", type=str, default='relu',
                        help='Types of activation: [relu, tanh, elu]')
    parser.add_argument("--factor", type=int, default=1,
                        help='factor to downsample images')
    

    # train vae options
    parser.add_argument('--vae_d', type=int, default=4)
    parser.add_argument('--vae_w', type=int, default=512)
    parser.add_argument('--vae_latent', type=int, default=32)
    parser.add_argument('--vae_kl_lambda', type=float, default=0.1)

    args = parser.parse_args()
    return args