# train on ibmq_santiago, test on ibmq_santiago, no batch norm, no quantize
python examples/train.py examples/configs/simple2cls/two36/train/noaddnoise/nonorm/ryrycx/n2b1/default.yml --gpu=0

# train on classic device, test on ibmq_santiago, no batch norm, no quantize
python examples/train.py examples/configs/simple2cls/two36/train/noaddnoise/nonorm/ryrycx/n2b1/clsTrainRealqcValid.yml --gpu=1

# train on ibmq_santiago, test on ibmq_santiago, add batch norm, no quantize
python examples/train.py examples/configs/simple2cls/two36/train/noaddnoise/bnormnolast/ryrycx/n2b1/default.yml --gpu=2

# train on classic device, test on ibmq_santiago, add batch norm, no quantize
python examples/train.py examples/configs/simple2cls/two36/train/noaddnoise/bnormnolast/ryrycx/n2b1/clsTrainRealqcValid.yml --gpu=3

# train on classic device, test on classic device, add batch norm, no quantize
python examples/train.py examples/configs/simple2cls/two36/train/noaddnoise/bnormnolast/ryrycx/n2b1/clsTrainClsValid.yml --gpu=4

#batch norm and quantize
python examples/train.py examples/configs/simple3cls/three012/train/noaddnoise/bnormnolastandquant/ryrycx/n2b1/clsTrainClsValid.yml --gpu=2

conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/nonorm/ryrycx/n2b1/ibmq_manila/clsTrainRealqcValid.yml --gpu=0
conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/nonorm/ryrycx/n2b1/ibmq_manila/realqcTrainRealqcValid.yml --gpu=0
conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/bnormnolast/ryrycx/n2b1/ibmq_manila/realqcTrainRealqcValid.yml --gpu=0


conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/nonorm/ryrycx/n2b1/ibmq_lima/clsTrainRealqcValid.yml --gpu=1
conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/nonorm/ryrycx/n2b1/ibmq_lima/realqcTrainRealqcValid.yml --gpu=1
conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/bnormnolast/ryrycx/n2b1/ibmq_lima/realqcTrainRealqcValid.yml --gpu=1


conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/nonorm/ryrycx/n2b1/ibmq_quito/clsTrainRealqcValid.yml --gpu=2
conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/nonorm/ryrycx/n2b1/ibmq_quito/realqcTrainRealqcValid.yml --gpu=2
conda activate qtm
python examples/train.py examples/configs/simple2cls_tiny2/two36/train/noaddnoise/bnormnolast/ryrycx/n2b1/ibmq_quito/realqcTrainRealqcValid.yml --gpu=2