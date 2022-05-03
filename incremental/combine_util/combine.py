import os
from argparse import ArgumentParser, Namespace
import pickle
import pathlib
import logging


def main(args):
    logging.basicConfig(filename='combined.log', level=logging.INFO)

    tar_list = list(pathlib.Path(r".").glob(r"*.tar"))
    logging.info('There are {} to be proccessed.'.format(str(tar_list)))
    for id, tar in enumerate(tar_list):
        os.system('tar xvf {}'.format(tar))
    
    os.system('mkdir combined')

    # find all files with .csv postfix recursively 
    csv_list = list(pathlib.Path(r".").glob(r"**/*.csv"))
    pkl_list = list(pathlib.Path(r".").glob(r"**/" + args.name + r".pkl"))

    # copy all csv file to combined directory
    for csv in csv_list:
        os.system('cp {} ./combined'.format(csv))

    logging.info('Copy {} files to combined.'.format(str(len(csv_list))))

    # merge pkl file
    logging.info('There are {} to be merged.'.format(str(pkl_list)))
    new_pkl = []
    print(pkl_list)
    for pkl in pkl_list:
        with open(pkl, 'rb') as f:
            new_pkl += pickle.load(f)
    
    logging.info('The length of merged {}.pkl is {}.'.format(args.name, str(len(new_pkl))))

    with open('./combined/' + args.name + '.pkl', 'wb') as f:
        pickle.dump(new_pkl, f)
    logging.info('Pickle dump to {}. complete.'.format('./combined/' + args.name + '.pkl'))


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--name", type=str, default='cifar10')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
