import argparse
import itertools
from scipy import stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth', type=str)
    parser.add_argument('--get', type=str)
    parser.add_argument('--truthloss', action='store_true')
    parser.add_argument('--getloss', action='store_true')

    args = parser.parse_args()

    truthinfo = 'Loss' if args.truthloss else 'Accuracy'

    cnt = 0
    truth = []
    with open(args.truth, 'r') as rfid:
        for line in rfid:
            if truthinfo in line:
                cnt += 1
                if cnt % 2:
                    truth.append(eval(line.split(' ')[-1]))


    getinfo = 'Loss' if args.getloss else 'Accuracy'

    cnt = 0
    get = []
    with open(args.get, 'r') as rfid:
        for line in rfid:
            if getinfo in line:
                cnt += 1
                if cnt % 2:
                    get.append(eval(line.split(' ')[-1]))


    # corrects = 0
    # cnt = 0
    # for comb in itertools.combinations_with_replacement(
    #         list(range(len(ground_truth_loss))), 2):
    #
    #     a = ground_truth_loss[comb[0]]
    #     b = ground_truth_loss[comb[1]]
    #     a_est = loss_all[comb[0]]
    #     b_est = loss_all[comb[1]]
    #     # if a < 0.5 or b < 0.5:
    #     #     continue
    #     cnt += 1
    #     if not (a >= b) ^ (a_est >= b_est):
    #         corrects += 1

    rho, p = stats.spearmanr(truth, get)

    print(f"spearman rho {rho}, p {p}")
