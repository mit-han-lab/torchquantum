from torchpack.utils.logging import logger
import argparse

import torch

import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=str)
    parser.add_argument('--noisy', type=str)
    parser.add_argument('--pdb', action='store_true')
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--path', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    print(args)

    file = "mnist.four0123.eval.device.real.opt2.noancilla.300_s18400.pt"
    if args.pdb:
        import pdb
        pdb.set_trace()

    if args.clean is None:
        clean_acts = torch.load(
            f"{args.path}.{args.arch}.default/activations/"
            f"mnist.four0123.eval.tq.300_s18400.pt")
        noisy_acts = torch.load(
            f"{args.path}.{args.arch}.default/activations/"
            f"{file.replace('device', args.device)}"
        )

    else:
        clean_acts = torch.load(args.clean)
        noisy_acts = torch.load(args.noisy)

    for noisy_act in noisy_acts:
        noisy = noisy_act['x_before_add_noise']
        noisy_act['x_all_norm'] = (noisy - noisy.mean()) / noisy.std()
        noisy_act['x_batch_norm'] = (noisy - noisy.mean(0)) / noisy.std(0)
        noisy_act['x_layer_norm'] = (noisy - noisy.mean(-1).unsqueeze(-1)) / \
            noisy.std(-1).unsqueeze(-1)

    for clean_act in clean_acts:
        clean = clean_act['x_before_add_noise']
        clean_act['x_all_norm'] = (clean - clean.mean()) / clean.std()
        clean_act['x_batch_norm'] = (clean - clean.mean(0)) / clean.std(0)
        clean_act['x_layer_norm'] = (clean - clean.mean(-1).unsqueeze(-1)) / \
            clean.std(-1).unsqueeze(-1)

    for k, (clean_act, noisy_act) in enumerate(zip(clean_acts, noisy_acts)):
        logger.info(f"Node {k}")
        for stage in ['x_before_add_noise', 'x_before_act_quant',
                      'x_all_norm', 'x_batch_norm', 'x_layer_norm']:
            logger.info(f"{stage}")
            diff = noisy_act[stage] - clean_act[stage]
            logger.info(f"clean mean {clean_act[stage].mean()}, "
                        f"std {clean_act[stage].std()}")
            logger.info(f"noisy mean {noisy_act[stage].mean()}, "
                        f"std {noisy_act[stage].std()}")

            logger.info(
                f"Error Per qubit: \n"
                f"Error Mean: {diff.mean(0).data.numpy()}, "
                f"Error Std: {diff.std(0).data.numpy()}\n"
                f"Error ALL: \n"
                f"Error Mean: {diff.mean()}, "
                f">0: {diff[clean_act[stage]>0].mean()}, "
                f"<0: {diff[clean_act[stage]<0].mean()}\n"
                f"Error Std: {diff.std()}, "
                f">0: {diff[clean_act[stage]>0].std()}, "
                f"<0: {diff[clean_act[stage]<0].std()}"
            )

            diff_abs_percent = diff.abs().mean() / clean_act[stage].abs(
                ).mean()
            logger.info(f"diff abs percent {diff_abs_percent}")

            diff_square_percent = diff.square().mean() / clean_act[
                stage].square().mean()
            logger.info(f"diff square percent {diff_square_percent}")

            if args.draw:
                plt.hist((clean_act[stage]).numpy().flatten(),
                         bins=50, alpha=0.5, label='clean')
                plt.hist((noisy_act[stage]).numpy().flatten(),
                         bins=50, alpha=0.5, label='noisy')
                plt.title(f'{stage}')
                plt.gca().legend()
                plt.savefig(
                    f"./examples/noise_script/plot/{args.path}.{args.arch}."
                    f"{args.device}.node{k}.{stage}.pdf")
                plt.close()
                plt.hist(diff.flatten().numpy(), bins=50, alpha=0.5,
                         label='error_all')
                plt.hist((diff[clean_act[stage]>0]),
                         bins=50, alpha=0.5, label='>0_all')
                plt.hist((diff[clean_act[stage]<0]),
                         bins=50, alpha=0.5, label='<0_all')
                plt.gca().legend()
                plt.savefig(
                    f"./examples/noise_script/plot/{args.path}.{args.arch}."
                    f"{args.device}.node{k}.{stage}.error.pdf")
                plt.close()






