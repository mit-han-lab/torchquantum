import pdb
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--pop_size', type=int)
    parser.add_argument('--parent_size', type=int)
    args = parser.parse_args()

    # pdb.set_trace()

    pop_size = args.pop_size
    keep_size = args.parent_size
    evo_loss_all = {}
    with open(f'./{args.dataset}.{args.name}_evo.csv') as fid:
        lines = fid.read().split('\n')[1:]
        for line in lines:
            if not line.strip():
                continue
            wall, step, loss = line.split(',')
            loss = eval(loss)
            step = eval(step)
            n_iter = step // pop_size
            if n_iter in evo_loss_all.keys():
                evo_loss_all[n_iter].append(loss)
            else:
                evo_loss_all[n_iter] = [loss]

    for n_iter, losses in evo_loss_all.items():
        evo_loss_all[n_iter] = sorted(losses)[:keep_size]

    rand_loss_all = {}
    with open(f'./{args.dataset}.{args.name}_rand.csv') as fid:
        lines = fid.read().split('\n')[1:]
        for line in lines:
            if not line.strip():
                continue
            wall, step, loss = line.split(',')
            loss = eval(loss)
            step = eval(step)
            n_iter = step // pop_size
            if n_iter in rand_loss_all.keys():
                rand_loss_all[n_iter].append(loss)
            else:
                rand_loss_all[n_iter] = [loss]

    for n_iter, losses in rand_loss_all.items():
        rand_loss_all[n_iter] = sorted(losses)[:keep_size]

    # print(evo_loss_all)
    # print(rand_loss_all)
    with open(f'./{args.dataset}.{args.name}.gen.csv', 'w') as wfid:
        for n_iter in evo_loss_all.keys():
            for evo_loss, rand_loss in zip(evo_loss_all[n_iter],
                                           rand_loss_all[n_iter]):
                wfid.write(f"{n_iter+1},{evo_loss},{n_iter+1},{rand_loss}\n")

    # print(loss_all)
