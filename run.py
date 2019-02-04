import sys
import os
import getopt
import shutil

import time
import datetime

from src.train import Worker


def reset_output_directory(remove_exist=True):
    """reset_output_directory

    :param remove_exist: If true, remove all existance folder. [default=True]
    """
    LOG_PATH = './logs'
    MODEL_PATH = './model'
    RENDER_PATH = './render'

    if remove_exist:
        if os.path.exists(LOG_PATH):
            shutil.rmtree(LOG_PATH, ignore_errors=False)
        if os.path.exists(MODEL_PATH):
            shutil.rmtree(MODEL_PATH, ignore_errors=False)
        if os.path.exists(RENDER_PATH):
            shutil.rmtree(RENDER_PATH, ignore_errors=False)

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(RENDER_PATH):
        os.makedirs(RENDER_PATH)


def parse_args(argv):
    """parse_args

    :param argv: Argument string on execution
    """
    def help():
        """help : output help string for exceptions"""
        print("SYNOPSIS: python run.py --epoch 300 --append-train False[True]")
        return

    try:
        # Append ':' for single character; append '=' for long string
        opts, args = getopt.getopt(argv, "h:", ["epoch=", "append-train=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        help()
        sys.exit(1)

    # Default Arguments
    new_train = True

    # Set Variables
    for opt, arg in opts:
        if opt == "--epoch":
            epoch = arg
            print(f'Epoch set to {arg}')
        elif opt == "--append-train":
            if arg:
                new_train = False
                print('Continue Training on Existing Weight')
            else:
                new_train = True
                print('Initiate Training')
        elif (opt == "-h") or (opt == "--help"):
            help()

    try:
        return {'epoch': epoch, 'new_train': new_train}
    except UnboundLocalError:
        print('Missing Arguments')
        help()
        sys.exit(1)


if __name__ == "__main__":
    par = parse_args(sys.argv[1:])
    epoch = par['epoch']

    reset_output_directory(par['new_train'])

    stime = time.time()
    sdate = datetime.datetime.now()
    print('Start Program ' + sdate.strftime("%Y-%m-%d %H:%M"))

    worker = Worker(epoch)
    worker.run()

    print(f'End {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Total Run Time : {time.time()-stime} sec')
