from libs import *

get_seed(1127802)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing {device}\n")
h = 1/201


def main():
    parser = argparse.ArgumentParser(description='Evalution')
    parser.add_argument('--model', type=str, default='uit', metavar='model',
                        help='evaluation model name, uit (integral transformer), uit-c3 (UIT with 3 channels) , ut (with traditional softmax normalization), hut (hybrid ut with linear attention), xut (cross-attention with hadamard product interaction), fno2d (Fourier neural operator 2d), unet (traditional UNet with CNN, big baseline, 33m params), unets (UNet with the same number of layers with U-integral transformer). default: uit)')
    parser.add_argument('--parts', nargs='+', default=[p for p in range(4, 7)],
                        help='parts of data used in training/evaluation. default: [4, 5, 6]')
    parser.add_argument('--plot-index', type=int, default=6, metavar='idx_draw',
                        help='the index of the inclusion to plot (default: 6)')
    parser.add_argument('--channels', type=int, default=1, metavar='num_chan',
                        help='the number of channels of feature maps (default: 1)')
    parser.add_argument('--noise', type=int, default=0, metavar='noise',
                        help='the noise level for eval (0, 5, 20) (default: 0)')
    parser.add_argument('--subsample', type=int, default=1, metavar='sample_scaling',
                        help='subsample scale, subsample=2 means (101,101) input (default: 1)')
    parser.add_argument('--batch-size', type=int, default=20, metavar='batch_size',
                        help='batch size for testing set (default: 20)')
    parser.add_argument('--no-grad-channel', action='store_true', default=False)
    parser.add_argument("--export-fig", action='store_true', default=False)
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

    weights_filename = config.weights_filename+".pt"
    with timer(f"\nLoading weights {weights_filename}"):
        model.load_state_dict(torch.load(
            os.path.join(MODEL_PATH, weights_filename)))
        model.to(device)

    valid_dataset = EITDataset(part_idx=args.parts,
                               file_type='h5',
                               noise=args.noise,
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

    metric_funcs = {"cross entropy": CrossEntropyLoss2d(regularizer=False, h=h),
                    "relative L2": L2Loss2d(regularizer=False, h=h),
                    "dice": SoftDiceLoss()}

    with timer(f"Evaluating"):
        val_results = validate_epoch_eit(
            model, metric_funcs, valid_loader, device)

    print(
        f"\nEvaluation result for {model.__class__.__name__} with {get_num_params(model)} params")
    for t in val_results.keys():
        print(f"{t}: {val_results[t]}")

    sample = next(iter(valid_loader))
    phi = sample['phi']
    gradphi = sample['gradphi']
    targets = sample['targets']
    grid = sample['grid']

    with torch.no_grad():
        model.eval()
        out_dict = model(phi.to(device), gradphi.to(
            device), grid=grid.to(device))

    preds = 0.5*(torch.tanh(out_dict['preds'].detach().cpu())+1)

    try:
        idx = args.plot_index
        pred = preds.numpy()[idx, ..., 0]
        target = targets.numpy()[idx, ..., 0]
        fig1 = showcontour(pred, width=300, height=300, template='plotly_white')
        fig2 = showcontour(target, width=300, height=300, template='plotly_white')
        if args.export_fig:
            fig1.write_image(os.path.join(FIG_PATH, 'preds.pdf'))
            fig2.write_image(os.path.join(FIG_PATH, 'targets.pdf'))
    except Exception as e:
        print(e.message)

if __name__ == "__main__":
    main()
