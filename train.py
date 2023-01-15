"""
TRAIN SKIP-ATTENTION GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly, skipattentionganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES
from datetime import datetime

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model


##
def main():
    """ Training
    """

    # Arguments
    opt = Options().parse()
    now = datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
    opt.log("Experiment Start:%s" % dt_string_start)
    # Load Data
    data = load_data(opt)
    exit()
    # Load Model
    model = load_model(opt, data)
    now = datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
    # Train Model
    model.train()
    now = datetime.now()
    dt_string_end = now.strftime("%d/%m/%Y %H:%M:%S")
    opt.log("Start:%s" % dt_string_start)
    opt.log("End:%s" % dt_string_end)


if __name__ == '__main__':
    main()
