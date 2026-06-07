from .infer_samples import main


if __name__ == "__main__":
    main(defaults={
        "data_path": "data/processed/nodes_train_norm.npz",
        "out_dir": "outputs/node_diffusion_norm_eval",
        "coord_scale": 160.0,
    })
