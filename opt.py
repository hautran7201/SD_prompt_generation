import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Data batch size')                                                 
    parser.add_argument("--epochs", type=int, default=1,
                        help='Epoch')   
    parser.add_argument("--learning_rate", type=float, default=4e-3,
                        help='Learning rate')
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help='Warm up step')    
    parser.add_argument('--model_out_dir', type=str, default='',
                        help='model output dir')                              
    parser.add_argument("--train_only", type=int, default=0,
                        help='Train model')    
    parser.add_argument("--eval_only", type=int, default=0,
                        help='Eval model')                             
    parser.add_argument("--infer_only", type=int, default=0,
                        help='Infer model')                             
    parser.add_argument("--infer_data", type=str, default='',
                        help='Infer data')  

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()                                                