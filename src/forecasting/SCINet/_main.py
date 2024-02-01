import torch
from experiment import Experiment
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='SCINet on financial datasets')
    # dataset                                                                                               # contam or cleaned
    parser.add_argument("--train_dataset_path",   type=str,       default="dataset/processed/Park/Residential/30_minutes/exp23/lf_contam",       help="Path to train dataset")
    parser.add_argument("--test_dataset_path",    type=str,       default="dataset/processed/Park/Residential/30_minutes/exp23/lf_test_clean",    help="Path to clean dataset for testing")
    # sequence
    parser.add_argument("--timesteps",            type=int,       default=48*7*2,     help="Number of timesteps")
    parser.add_argument("--sequence_split",       type=float,     default=0.5,      help="Ratio of input to target (forecasting horizon) split")
    parser.add_argument("--nbr_var",              type=int,       default=1,        help="Number of variables")
    # model parameters
    parser.add_argument("--hidden_size",          type=int,       default=128,      help="Hidden size of the model")
    parser.add_argument("--num_grulstm_layers",   type=int,       default=1,        help="Number of GRU/LSTM layers")
    parser.add_argument("--fc_units",             type=int,       default=16,       help="Number of fully connected units")
    # training
    parser.add_argument("--epochs",               type=int,       default=30,      help="Number of epochs")
    parser.add_argument("--patience",             type=int,       default=50,       help="Patience for early stopping")
    parser.add_argument("--batch_size",           type=int,       default=32,       help="Batch size")
    parser.add_argument("--lr",                   type=float,     default=1e-3,     help="Learning rate")
    parser.add_argument("--seed",                 type=int,       default=0)
    parser.add_argument("--checkpoint_path",      type=str,       default="src/forecasting/checkpoint.pt",       help="Path to save checkpoint")
    # visualization
    parser.add_argument("--n_plots",              type=int,       default=32,                                    help="Number of plots")
    parser.add_argument("--save_plots_path",      type=str,       default="results/forecasting/contam",          help="Path to save plots")
    parser.add_argument("--results_file",         type=str,       default="results/results.txt",                 help="Path to file to save results in")
    
    # =================== SCINet model parameters ===================

    ### -------  input/output length settings --------------                                                                            
    parser.add_argument('--window_size', type=int, default=48*7*2, help='input length')
    parser.add_argument('--horizon', type=int, default=48*7*2//2, help='prediction length')


    ### -------  training settings --------------  
    parser.add_argument('--save', type=str, default='model/model.pt', help='path to save the final model')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--L1Loss', type=bool, default=True)
    parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
    parser.add_argument('--lradj', type=int, default=2,help='adjust learning rate')
    parser.add_argument('--save_path', type=str, default='exp/financial_checkpoints/')
    parser.add_argument('--model_name', type=str, default='SCINet')

    ### -------  model settings --------------  
    parser.add_argument('--hidden-size', default=1.0, type=float, help='hidden channel of module')# H, EXPANSION RATE
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=5, type=int, help='kernel size')#k kernel size
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type = bool , default=False)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--num_decoder_layer', type=int, default=1)
    parser.add_argument('--stacks', type=int, default=1)
    parser.add_argument('--long_term_forecast', action='store_true', default=False)
    parser.add_argument('--RIN', type=bool, default=False)
    parser.add_argument('--decompose', type=bool,default=False)
    parser.add_argument('--concat_len', type=int, default=24*6)
    parser.add_argument('--single_step', type=int, default=0, help='only supervise the final setp')
    parser.add_argument('--single_step_output_One', type=int, default=0, help='only output the single final step')
    parser.add_argument('--lastWeight', type=float, default=0.5,help='Loss weight lambda on the final step')

    args = parser.parse_args()
    if not args.long_term_forecast:
        args.concat_len = args.window_size - args.horizon
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    exp = Experiment(args)
    exp.train()
