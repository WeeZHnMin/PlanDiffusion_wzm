from .infer_samples import main


if __name__ == "__main__":
    main(defaults={
        "data_path": "data/processed/nodes_train.npz",
        "out_dir": "outputs/node_diffusion_raw_eval",
        "coord_scale": 1.0,
    })
