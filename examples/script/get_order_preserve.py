import argparse
import itertools


if __name__ == '__main__':
    ground_truth_loss = [
        1.251312613,
        1.206900716,
        1.217935443,
        1.076185346,
        1.052040577,
        1.245998144,
        1.128251195,
        1.165869713,
        1.011000872,
        0.9784647226,
        1.229257464,
        1.065639853,
        1.114256978,
        0.9727173448,
        0.9656341076,
        1.228745103,
        1.069212556,
        1.113019466,
        0.9471424222,
        0.91166085,
        1.227254033,
        1.05306685,
        1.086424232,
        0.9468999505,
        0.9130370021,
        1.227954745,
        1.054544806,
        1.087273598,
        0.9213072062,
        0.9069637656,
        1.234271526,
        1.049979568,
        1.078618169,
        0.9178112149,
        0.8910576701,
        1.228218913,
        1.04969728,
        1.077585459,
        0.9398220181,
        0.8893611431,
        1.172919869,
        1.174831748,
        1.190154433,
        1.09282589,
        1.179687142,
        1.095748186,
        1.009546638,
        1.039857149,
        1.023582697,
        1.061262965,
        1.02767837,
        1.061795712,
        1.000618219,
        1.015908122,
        0.9676553011,
        0.968763411,
        0.9473308325,
        0.9578924179,
        0.9661900401,
        0.9710415006,
        0.9486585855,
        0.9542745352,
        0.9456375241,
        0.9971755743,
        0.9363359213,
        0.9315931201,
        0.9280217886,
        0.9378393888,
        0.9523043633,
        0.9223439693,
        0.9052116275,
        0.9448692799,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--supernet', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    print(args.supernet)

    cnt = 0
    loss_all = []
    with open(f'logs/super/eval_subnet_tq_ratiorand_'
              f'insuper_{args.supernet}.txt',
              'r') as rfid:
        for line in rfid:
            if 'Loss' in line:
                cnt += 1
                if cnt % 2:
                    loss_all.append(eval(line.split(' ')[-1]))

    corrects = 0
    cnt = 0
    for comb in itertools.combinations_with_replacement(
            list(range(len(ground_truth_loss)-13)), 2):

        a = ground_truth_loss[comb[0]]
        b = ground_truth_loss[comb[1]]
        a_est = loss_all[comb[0]]
        b_est = loss_all[comb[1]]
        # if a > 1 or b > 1:
        #     continue
        cnt += 1
        if not (a > b) ^ (a_est > b_est):
            corrects += 1

    print(f"Rate: {corrects / cnt}")
