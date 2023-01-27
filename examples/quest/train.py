import argparse
import pdb
import random
import sys

import core.datasets.builder as builder
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from core.datasets.trainer import trainer
from torch_geometric.loader import DataLoader
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from utils.load_data import load_data_and_save


def main() -> None:
    # load_data_and_save("huge.data")
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    configs.evalmode = False

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", metavar="FILE", help="config file")
    parser.add_argument("--load", action="store_true", help="config file")

    args, opts = parser.parse_known_args()

    configs.load(f"exp/{args.exp_name}/config.yaml", recursive=True)
    configs.update(opts)
    configs.exp_name = args.exp_name
    if configs.device == "gpu":
        device = torch.device("cuda")
    elif configs.device == "cpu":
        device = torch.device("cpu")

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)
    args.run_dir = auto_set_run_dir()

    logger.info(" ".join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + "\n" + f"{configs}")

    model = builder.make_model()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Size: {total_params}")
    dataflow = {}
    if not args.load:
        dataset = builder.make_dataset()
        for split in ["train", "valid", "test"]:
            dataflow[split] = DataLoader(
                dataset.get_data(device, split), batch_size=configs.batch_size
            )

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    my_trainer = trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataflow,
    )
    if not args.load:
        my_trainer.train()
        my_trainer.test()
        my_trainer.saveall()
    my_trainer.loadall()
    my_trainer.curve()
    my_trainer.scatter()


if __name__ == "__main__":
    main()
