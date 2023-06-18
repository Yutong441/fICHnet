if __name__ == "__main__":
    import argparse
    from fICHnet.run_model import predict_all

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_paths', type=str,
        help="path to a text file containing absolute path of all the images")
    parser.add_argument(
        '--model_path', type=str,
        help="path containing model weight")
    parser.add_argument(
        '--save_dir', type=str,
        help="where to save the prediction output")
    parser.add_argument('--device', type=str, default="cpu")
    args = parser.parse_args()
    predict_all(args.img_paths, args.model_path, args.save_dir, args.device)
