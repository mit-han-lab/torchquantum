if __name__ == '__main__':
    with open('logs/sfsuper/eval_subnet_suc_ratiorand.txt') as rfid:
        for line in rfid:
            if 'Success' in line:
                print(eval(line.split(' ')[-1]))
