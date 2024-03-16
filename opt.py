import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='',
                        help='Config file path')
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