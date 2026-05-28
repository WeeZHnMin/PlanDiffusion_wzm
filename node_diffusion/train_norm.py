from .train import main


if __name__ == "__main__":
    main(defaults={
        "data_path": "data/processed/nodes_train_norm.npz",
        "save_dir": "checkpoints/node_diffusion_norm",
    })
