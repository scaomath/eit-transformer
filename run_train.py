from libs import *

get_seed(1127802)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing {device}\n")
h = 1/201

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--model', type=str, default='unets', metavar='model',
                        help='training model name, uit (integral transformer), ut (with traditional softmax normalization), hut (hybrid ut with linear attention), xut (cross-attention with hadamard product interaction), fno2d (Fourier neural operator 2d), unet (traditional UNet with CNN, big baseline, 33m params), unets (UNet with the same number of layers with U-integral transformer). default: unets)')
    parser.add_argument('--parts', nargs='+', default=[p for p in range(4, 7)], 
                        help='parts of data used in training/evaluation. default: [4, 5, 6]')
    parser.add_argument('--plot-index', type=int, default=6, metavar='idx_draw',
                        help='the index of the inclusion to plot (default: 6)')
    parser.add_argument('--channels', type=int, default=1, metavar='num_chan',
                        help='the number of channels of feature maps (default: 1)')
    parser.add_argument('--subsample', type=int, default=1, metavar='sample_scaling',
                        help='subsample scale, subsample=2 means (101,101) input (default: 1)')
    parser.add_argument('--batch-size', type=int, default=10, metavar='batch_size',
                        help='batch size for testing set (default: 10)')
    parser.add_argument('--epochs', type=int, default=50, metavar='epochs',
                        help='number of epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=15, metavar='patience',
                        help='early stopping epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='learning_rate',
                        help='maximum learning rate (default: 1e-3)')
    parser.add_argument('--no-grad-channel', action='store_true', default=False)
    args = parser.parse_args()

    config = load_yaml(r'./configs.yml', key=args.model)
    print("="*10+"Model setting:"+"="*10)
    for a in config.keys():
        if not a.startswith('__'):
            print(f"{a}: {config[a]}")
    print("="*33)

    if args.model in ["uit", "uit-c3", "uit-c", "ut", "xut"]:
        from libs.ut import UTransformer
        model = UTransformer(**config)
    elif args.model in ["hut"]:
        from libs.hut import HybridUT
        model = HybridUT(**config)
    elif args.model in ["fno2d", "fno2d-big"]:
        from libs.fno import FourierNeuralOperator
        model = FourierNeuralOperator(**config)
    elif args.model in ["unet", "unet-small"]:
        from libs.unet import UNet
        model = UNet(**config)
    else:
        raise NotImplementedError
    
    print(f"\nTraining for {model.__class__.__name__} with {get_num_params(model)} params\n")
    model.to(device);

    train_dataset = EITDataset(part_idx=args.parts,
                           file_type='h5',
                           subsample=args.subsample,
                           channel=args.channels,
                           return_grad=not args.no_grad_channel,
                           online_grad=False,
                           train_data=True,)
    train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    valid_dataset = EITDataset(part_idx=args.parts,
                               file_type='h5',
                               channel=args.channels,
                               return_grad=not args.no_grad_channel,
                               online_grad=False,
                               subsample=args.subsample,
                               train_data=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                           div_factor=1e3, final_div_factor=1e4,
                           steps_per_epoch=len(train_loader), 
                           pct_start=0.2, epochs=args.epochs)

    loss_func = CrossEntropyLoss2d(regularizer=False, h=h, gamma=0.1)
    metric_func = L2Loss2d(regularizer=False)

    result = run_train(model, loss_func, metric_func,
                    train_loader, valid_loader,
                    optimizer, scheduler,
                    train_batch=train_batch_eit,
                    validate_epoch=validate_epoch_eit,
                    epochs=args.epochs,
                    patience=args.patience,
                    model_name=config.weights_filename+".pt",
                    result_name=config.weights_filename+".pkl",
                    tqdm_mode='batch',
                    mode='min',
                    device=device)
    print("Training done.")

if __name__ == "__main__":
    main()
