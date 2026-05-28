from .preprocess_nodes import main


if __name__ == "__main__":
    main([
        "--normalize",
        "--scale",
        "160.0",
        "--output",
        "data/processed/nodes_train_norm.npz",
    ])
