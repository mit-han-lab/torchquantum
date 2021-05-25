from torchpack.utils.logging import logger
import argparse
from matplotlib.ticker import FormatStrFormatter

import torch

import matplotlib.pyplot as plt
import os
import numpy as np


import numpy as np
from scipy.ndimage.filters import gaussian_filter

def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    cond = (p != 0)

    return np.sum(np.where(cond, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=40, sigma=1):

    mini = min(a.min(), b.min())
    maxi = max(a.max(), b.max())

    ahist, bhist = (np.histogram(a, bins=nbins,
                                 range=(mini, maxi),
                                 )[0],
                    np.histogram(b, bins=nbins,
                                 range=(mini, maxi),
                                 )[0])


    ahist = ahist / ahist.sum()
    bhist = bhist / bhist.sum()
    # asmooth = ahist
    # bsmooth = bhist

    asmooth, bsmooth = (gaussian_filter(ahist, sigma),
                        gaussian_filter(bhist, sigma))


    return kl(asmooth, bsmooth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=str)
    parser.add_argument('--noisy', type=str)
    parser.add_argument('--pdb', action='store_true')
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--drawfour', action='store_true')
    parser.add_argument('--path', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    # parser.add_argument('--')

    args = parser.parse_args()
    print(args)

    valid = '_valid' if args.valid else ''

    last_step_dict = {
        'mnist': {
            'four0123': 18400,
            'two36': 9000,
        },
        'fashion': {
            'four0123': 18000,
            'two36': 9000,
        },
        'vowel': {
            'four0516': 10400,
        }
    }

    act_quant = '.act_quant'



    if args.pdb:
        import pdb
        pdb.set_trace()

    if args.clean is None:
        file = f"{args.dataset}" \
               f".{args.name}.eval.{args.device}.real.opt2.noancilla.300_s" \
               f"{last_step_dict[args.dataset][args.name]}" \
               f"{valid}.pt"
        clean_acts = torch.load(
            f"{args.path}.{args.arch}.default/activations/"
            f"{args.dataset}.{args.name}.eval.tq{act_quant}"
            f".300_s{last_step_dict[args.dataset][args.name]}{valid}.pt")
        noisy_acts = torch.load(
            f"{args.path}.{args.arch}.default/activations/"
            f"{file}"
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

    # separators = np.linspace(-1, 1, 6)
    separators = [1]

    for k, (clean_act, noisy_act) in enumerate(zip(clean_acts, noisy_acts)):
        if k == len(clean_acts) - 1:
            continue
        logger.info(f"Node {k}")
        for stage in ['x_before_add_noise',
                      'x_before_act_quant',
                      # 'x_all_norm',
                      # 'x_batch_norm',
                      # 'x_layer_norm'
                      ]:
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

            for kk in range(len(separators) - 1):
                idx = (separators[kk] <= clean_act[stage]) * \
                      (clean_act[stage] < separators[kk + 1])
                logger.info(
                    f"Error [{separators[kk], separators[kk + 1]}): \n"
                    f"Error Per qubit: \n"
                    f"Error Mean: {diff[idx].mean(0).data.numpy()}, "
                    f"Error Std: {diff[idx].std(0).data.numpy()}\n"
                    f"Error ALL: \n"
                    f"Error Mean: {diff[idx].mean()}, "
                    f"Error Std: {diff[idx].std()}, "
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

                for kk in range(len(separators) - 1):
                    idx = (separators[kk] <= clean_act[stage]) * \
                          (clean_act[stage] < separators[kk + 1])
                    if idx.size():
                        plt.hist((diff[idx]).flatten().numpy(),
                                 bins=50,
                                 alpha=0.5,
                                 label=f'[{separators[kk]:.3f}, '
                                       f'{separators[kk + 1]:.3f})')

                plt.gca().legend()
                plt.savefig(
                    f"./examples/noise_script/plot/{args.path}.{args.arch}."
                    f"node{k}.{stage}.error.pdf")
                plt.close()

            # Compute SNR
            snr_all = clean_act[stage].square().mean() / diff.square().mean()
            logger.info(f"SNR All: {snr_all}")
            snr_qubits = []

            for qubit in range(4):
                snr_q = clean_act[stage][:, qubit].square().mean() / \
                        diff[:, qubit].square().mean()
                snr_qubits.append(snr_q)
                logger.info(f"SNR Qubit {qubit}: {snr_q}")

            # compute KL
            # KL_all = smoothed_hist_kl_distance(
            #     noisy_act[stage].flatten().numpy(),
            #     clean_act[stage].flatten().numpy(),
            #     nbins=bins,
            #     )
            #
            # logger.info(f"KL All: {KL_all}")
            #
            # KL_qubits = []
            # for qubit in range(4):
            #     KL_q = smoothed_hist_kl_distance(
            #         noisy_act[stage][:, qubit].flatten().numpy(),
            #         clean_act[stage][:, qubit].flatten().numpy(),
            #         nbins=bins)
            #     # KL_q = clean_act[stage][:, qubit].square().mean() / \
            #     KL_qubits.append(KL_q)
            #     logger.info(f"KL Qubit {qubit}: {KL_q}")




            if args.drawfour:
                name = args.noisy.split('/')[-1]

                #  heatmap
                snr_singles = (clean_act[stage].square() / diff.square(
                )).permute(1, 0)

                fig, ax = plt.subplots()
                # ax = fig.add_subplot(111)
                cax = ax.matshow(snr_singles[:, :50], cmap='Blues')
                # cax = ax.matshow(softmax, cmap='Reds')

                # print(softmax[1:-1, 1:-1].sum(dim=-2))

                cax.set_clim(0, 8)
                fig.colorbar(cax, orientation="horizontal", pad=0.2)
                font_tick = {'family': 'Arial',
                        'weight': 'normal',
                        'size': 10,
                        }

                ax.set_xticklabels(ax.get_xticks(), font_tick)
                ax.set_yticklabels(ax.get_yticks(), font_tick)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))


                # plt.imshow(snr_singles, cmap='hot', interpolation='nearest')
                plt.savefig(
                    f"./examples/noise_script/plot/four/{name}."
                    f"node{k}.{stage}.heatmap.pdf",
                    bbox_inches='tight',
                    pad_inches=0)
                plt.close()


                bins = 40
                for qubit in range(4):
                    mini = min(clean_act[stage][:, qubit].min(),
                               noisy_act[stage][:, qubit].min()).item()
                    maxi = max(clean_act[stage][:, qubit].max(),
                               noisy_act[stage][:, qubit].max()).item()
                    plt.figure(figsize=(4, 4*0.618))
                    plt.hist((clean_act[stage][:, qubit]).numpy().flatten(),
                         bins=bins, alpha=0.7, label='clean',
                             color=[0/255, 129/255, 204/255],
                             range=[mini, maxi]
                    )
                    font = {'family': 'Arial',
                            'weight': 'normal',
                            'size': 20,
                            }

                    plt.hist((noisy_act[stage][:, qubit]).numpy().flatten(),
                         bins=bins, alpha=0.7, label='noisy',
                             color=[248/255, 182/255, 45/255],
                             range=[mini, maxi])
                    plt.text(0.25, 0.8,
                             f"Qubit {qubit}\nSNR={snr_qubits[qubit]:.2f}",
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=plt.gca().transAxes,
                             fontdict=font
                             )
                    # plt.title(f'{stage}')
                    # plt.gca().legend()
                    frame1 = plt.gca()
                    frame1.axes.get_xaxis().set_visible(False)
                    frame1.axes.get_yaxis().set_visible(False)

                    # plt.gca().set_axis_off()
                    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    # hspace = 0, wspace = 0)
                    # plt.margins(0,0)
                    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

                    plt.savefig(
                        f"./examples/noise_script/plot/four/{name}."
                        f"node{k}.{stage}.qubit{qubit}.pdf",
                        bbox_inches='tight',
                        pad_inches=0)

                    plt.close()


                    # plt.hist(diff[:, qubit].flatten().numpy(), bins=50,
                    #          alpha=0.5, label='error_all')
                    # plt.hist((diff[:, qubit][clean_act[stage][:, qubit]>0]),
                    #          bins=bins, alpha=0.5, label='>0_all')
                    # plt.hist((diff[:, qubit][clean_act[stage][:, qubit]<0]),
                    #          bins=bins, alpha=0.5, label='<0_all')

                    # for kk in range(len(separators) - 1):
                    #     idx = (separators[kk] <= clean_act[stage]) * \
                    #           (clean_act[stage] < separators[kk + 1])
                    #     if idx.size():
                    #         plt.hist((diff[idx]).flatten().numpy(),
                    #                  bins=50,
                    #                  alpha=0.5,
                    #                  label=f'[{separators[kk]:.3f}, '
                    #                        f'{separators[kk + 1]:.3f})')

                    # plt.gca().legend()
                    # plt.savefig(
                    #     f"./examples/noise_script/plot/four/{name}."
                    #     f"node{k}.{stage}.qubit{qubit}.error.pdf")
                    # plt.close()






