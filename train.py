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
    # Load Data
    data = load_data(opt)
    # Load Model
    model = load_model(opt, data)
    now = datetime.now()
    dt_string1 = now.strftime("Start: %d/%m/%Y %H:%M:%S")
    # Train Model
    model.train()
    now = datetime.now()
    dt_string = now.strftime("Start: %d/%m/%Y %H:%M:%S")
    print("Start:", dt_string1)
    print("End:", dt_string)


if __name__ == '__main__':
    main()
