import argparse

def str2bool(str):
	return True if str.lower() == 'true' else False


def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-net', type=str, required=True, help='sam,sam_adapter,sam_fineTuning,PromptVit')
    parser.add_argument('-exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('-data_path', type=str, required=True, help='The path of segmentation data')
    parser.add_argument('-sam_ckpt', type=str, required=True, help='SAM checkpoint address')

    # General settings
    parser.add_argument('-mod', type=str, required=False, help='Mod type: seg, cls, val_ad')
    parser.add_argument('-dataset', type=str, default='SHA', help='Dataset name')
    parser.add_argument('-data_name', type=str, default='SHA',help='SHA')
    parser.add_argument('-save_path', type=str, default='model_checkpoint/wjj_adpt/', help='Path to save results')
    parser.add_argument('-mode', type=str, choices=['train', 'test'], help='Mode: train or test')

    # Model configurations
    parser.add_argument('-model_type', type=str, default="vit_h", help='SAM ViT model type')
    parser.add_argument('-baseline', type=str, default='unet', help='Baseline net type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='Segmentation net type')
    parser.add_argument('-prompt_approach', type=str, default="points_grids",
                        help='Prompt approach: random_click or points_grids')
    parser.add_argument('-image_encoder_configuration', type=int, nargs='+',
                        default=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                 3, 3, 3],
                        help='Image encoder configuration: 0: original SAM, 1: space adapter, 2: MLP adapter, 3: space + MLP adapter')
    parser.add_argument('-fine_tuning_configuration', type=int, nargs='+',
                        default=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0],
                        help="1: doesn't freeze the specific block, 0: freeze the block")
    parser.add_argument('-image_size', type=int, default=1024, help='Input image size')
    parser.add_argument('-out_size', type=int, default=256, help='Output size')
    parser.add_argument('-patch_size', type=int, default=2, help='Patch size')
    parser.add_argument('-dim', type=int, default=512, help='Dimension size')
    parser.add_argument('-depth', type=int, default=1, help='Depth')
    parser.add_argument('-heads', type=int, default=16, help='Number of heads')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='MLP dimension')

    # Token settings
    parser.add_argument('-NUM_TOKENS', type=int, default=64, help='Number of tokens to add (multiple of 64)')
    parser.add_argument('-LOCATION', type=str, default='prepend', help='Token location')
    parser.add_argument('-INITIATION', type=str, default='random', help='Token initiation')
    parser.add_argument('-DROPOUT', type=float, default=0.0, help='Token dropout')
    parser.add_argument('-PROJECT', type=int, default=-1, help='Token project')
    parser.add_argument('-token_output_type', type=str, default='slice', help='Token output type: slice/linear')
    parser.add_argument('-PROMPT_DEEP', type=str2bool, default=False, help='Whether to use shallow or deep prompt')
    parser.add_argument('-deep_token_block_configuration', type=int, nargs='+',
                        default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1],
                        help='Specify which block (0 or 1) uses deep token')

    # Training configurations
    parser.add_argument('-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-epoch_ini', type=int, default=1, help='Start epoch')
    parser.add_argument('-batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('-warm', type=int, default=1, help='Warm-up training phase')
    parser.add_argument('-lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='Implicit learning rate')
    parser.add_argument('-weights', type=str, default=0, help='The weights file to test')
    parser.add_argument('-base_weights', type=str, default=0, help='Baseline weights')
    parser.add_argument('-sim_weights', type=str, default=0, help='Sim weights')
    parser.add_argument('-ckpt', type=str, default='', help='my pretrain path')

    # GPU settings
    parser.add_argument('-gpu', type=bool, default=True, help='Use GPU or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='GPU device ID')
    parser.add_argument('-sim_gpu', type=int, default=0, help='Split SIM to this GPU')
    parser.add_argument('-distributed', default='none', type=str, help='multi GPU ids to use')

    # Evaluation and visualization
    parser.add_argument('-vis', type=int, default=1, help='Visualization')
    parser.add_argument('-val_freq', type=int, default=1, help='Validation frequency')
    parser.add_argument('-reverse', type=bool, default=False, help='Adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='Pretrain or not')
    parser.add_argument('-min_mask_region_area', type=int, default=0, help='Minimum mask region area')
    parser.add_argument('-type', type=str, default='map', help='Condition type: ave, rand, rand_map')

    # 3D data processing
    parser.add_argument('-thd', type=bool, default=False, help='3D data or not')
    parser.add_argument('-chunk', type=int, default=96, help='Crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4, help='Number of positive/negative samples')
    parser.add_argument('-roi_size', type=int, default=96, help='ROI resolution')
    parser.add_argument('-evl_chunk', type=int, default=None, help='Evaluation chunk')

    opt = parser.parse_args()
    return opt

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-net', type=str, required=True, help='net type')
#     parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
#     parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
#     parser.add_argument('-mod', type=str, required=False, help='mod type:seg,cls,val_ad')
#     parser.add_argument('-exp_name', type=str, required=True, help='net type')
#     parser.add_argument('-prompt_approach', type=str, required=False, default="points_grids", help='the prompt approach: random_click or '
#                                                                           'points_grids')
#     parser.add_argument('-image_encoder_configuration',type=int, required=False, nargs='+',
#                         default=[3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3],
#                         help='image encoder configuration: 0: original sam. 1: space adapter. 2:MLP adapter. '
#                              '3: space adapter + MLP adapter. ')
#     parser.add_argument('-sam_vit_model', type=str, required=False, default="h", help='')
#     parser.add_argument('-CryoPPP_image_sice', type=int, required=False, nargs='+', default=[4096, 4096], help='')
#     parser.add_argument('-Groundtruth_path', type=str, required=False, help='')
#     # token prompt
#     parser.add_argument('-NUM_TOKENS', type=int, default=64, required=False, help='token number to add(Integer multiple of 64)')
#     parser.add_argument('-LOCATION', type=str, default='prepend', required=False, help='token LOCATION')
#     parser.add_argument('-INITIATION', type=str, default='random', required=False, help='token INITIATION')
#     parser.add_argument('-DROPOUT', type=float, default=0.0, required=False, help='token DROPOUT')
#     parser.add_argument('-PROJECT', type=int, default=-1, required=False, help='token PROJECT')
#     parser.add_argument('-token_output_type', type=str, default='slice', required=False, help='slice/linear')
#     parser.add_argument('-PROMPT_DEEP', type=str2bool, default=False, required=False, help='whether to use shallow prompt or deep prompt')
#     parser.add_argument('-deep_token_block_configuration', type=int, required=False, nargs='+',
#                         default=[1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1],
#                         help='specify which block(31 block can use deep token, the first block use shallow token in default source code) add deep token : '
#                              '0: without deep token. 1: add deep token. ')
#     parser.add_argument('-iteration', type=int, default=0, required=False, help='whether to use iteration')
#     parser.add_argument('-token_method', type=str, default="new", required=False, help='select token method')
#     #
#     parser.add_argument('-fine_tuning_configuration',type=int, required=False, nargs='+',
#                         default=[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,0],
#                         help="1: doesn't freeze the specific block, 0: freeze the block")
#     parser.add_argument('-min_mask_region_area', type=int,default=0, required=False, help='min mask region area')
#     parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
#     parser.add_argument('-vis', type=int, default=1, help='visualization')
#     parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
#     parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
#     parser.add_argument('-val_freq',type=int,default=1,help='interval between each validation')
#     parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
#     parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
#     parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
#     parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
#     parser.add_argument('-image_size', type=int, default=1024, help='image_size')
#     parser.add_argument('-out_size', type=int, default=256, help='output_size')
#     parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
#     parser.add_argument('-dim', type=int, default=512, help='dim_size')
#     parser.add_argument('-depth', type=int, default=1, help='depth')
#     parser.add_argument('-heads', type=int, default=16, help='heads number')
#     parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
#     parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
#     parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
#     parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
#     parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
#     parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
#     parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
#     parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
#     parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
#     parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
#     parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
#     parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
#     parser.add_argument('-dataset', default='CryoPPP' ,type=str,help='dataset name')
#     parser.add_argument('-sam_ckpt', default=None , help='sam checkpoint address')
#     parser.add_argument('-thd', type=bool, default=False , help='3d or not')
#     parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
#     parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
#     parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
#     parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
#     parser.add_argument(
#     '-data_path',
#     type=str,
#     required=True,
#     default='../dataset',
#     help='The path of segmentation data')
#     # '../dataset/RIGA/DiscRegion'
#     # '../dataset/ISIC'
#
#     opt = parser.parse_args()
#
#     return opt
