import argparse
import os
import torch
import sys
import numpy as np
import pandas as pd
import random

pd.options.display.float_format = '{:,.6f}'.format
pd.set_option('display.width', 160)
pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser(description='Analize results in csv files')

parser.add_argument('-p', '--path', default="", type=str, help='Path for the experiments to be analized')
parser.set_defaults(argument=True)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def main():

    DATASETS = ['cifar10', 'cifar100', 'imagenet1k']
    MODELS = ['densenetbc100', 'resnet34', 'wideresnet2810','resnet18'] 
    LOSSES = ['softmax_std_std_std', 'isomax_std_std_std', 'isomaxplus_std_std_std','dismax_std_std_std', 'dismax_std_std_noreg','dismax_std_std_0.5']

    print(DATASETS)
    print(MODELS)
    print(LOSSES)

    args = parser.parse_args()
    path = os.path.join("experiments", args.path)
    if not os.path.exists(path):
        sys.exit('You should pass a valid path to analyze!!!')

    #print("\n#####################################")
    #print("########## FINDING FILES ############")
    #print("#####################################\n")
    list_of_files = []
    file_names_dict_of_lists = {}
    for (dir_path, dir_names, file_names) in os.walk(path):
        for filename in file_names:
            if filename.endswith('.csv') or filename.endswith('.npy') or filename.endswith('.pth'):
                if filename not in file_names_dict_of_lists:
                    file_names_dict_of_lists[filename] = [os.path.join(dir_path, filename)]
                else:
                    file_names_dict_of_lists[filename] += [os.path.join(dir_path, filename)]
                list_of_files += [os.path.join(dir_path, filename)]
    #for key in file_names_dict_of_lists:
    #    print(key)

    #print("\n#####################################")
    #print("######## TABLE: RAW RESULTS #########")
    #print("#####################################\n")
    #data_frame_list = []
    #for file in file_names_dict_of_lists['results_raw.csv']:
    #    data_frame_list.append(pd.read_csv(file))
    #raw_results_data_frame = pd.concat(data_frame_list)
    #print(raw_results_data_frame[:30])

    print("\n###################################################")
    print("###################################################")
    print("###################################################")
    print("###### TABLE: BEST ACCURACIES ON CLEAN DATA #######")
    print("###################################################")
    print("###################################################")
    print("###################################################\n")
    data_frame_list = []
    for file in file_names_dict_of_lists['results_best.csv']:
        data_frame_list.append(pd.read_csv(file))
    best_results_data_frame = pd.concat(data_frame_list)
    best_results_data_frame.to_csv(os.path.join(path, 'all_results_best.csv'), index=False)

    for data in DATASETS:
        for model in MODELS:
            print("\n##########################")
            print(data)
            print(model)
            df = best_results_data_frame.loc[
                best_results_data_frame['DATA'].isin([data]) &
                best_results_data_frame['MODEL'].isin([model])
            ]
            df = df.rename(columns={'VALID MAX_PROBS MEAN': 'MAX_PROBS', 'VALID ENTROPIES MEAN': 'ENTROPIES',
                                    'VALID INTRA_LOGITS MEAN': 'INTRA_LOGITS', 'VALID INTER_LOGITS MEAN': 'INTER_LOGITS'})
            df = df.groupby(['LOSS'], as_index=False)[[
                'TRAIN LOSS', 'TRAIN ACC1','VALID LOSS', 'VALID ACC1',
                'ENTROPIES',]].agg(['mean','std','count'])
            df = df.sort_values([('VALID ACC1','mean')], ascending=False)#.drop_duplicates(["LOSS"])
            print(df)
            print("##########################\n")


    print("\n#####################################")
    print("#####################################")
    print("#####################################")
    print("######## TABLE: ODD METRICS #########")
    print("#####################################")
    print("#####################################")
    print("#####################################\n")
    data_frame_list = []
    for file in file_names_dict_of_lists['results_odd.csv']:
        data_frame_list.append(pd.read_csv(file))
    best_results_data_frame = pd.concat(data_frame_list)
    best_results_data_frame.to_csv(os.path.join(path, 'all_results_odd.csv'), index=False)

    for data in DATASETS:
        for model in MODELS:
            print("\n#####################################################################################")
            print(data)
            print(model)
            df = best_results_data_frame.loc[
                best_results_data_frame['IN-DATA'].isin([data]) &
                best_results_data_frame['MODEL'].isin([model]) &
                best_results_data_frame['SCORE'].isin(["MPS","ES","MDS","MMLS","MMLES","MLES","MMLEPS"]) &
                best_results_data_frame['OUT-DATA'].isin(['svhn','lsun_resize','imagenet_resize','cifar10', 'cifar100', 'imagenet-o'])]

            #############################################################################################################################
            #############################################################################################################################
            df = df[['MODEL','IN-DATA','LOSS','SCORE','EXECUTION','OUT-DATA','TNR','AUROC','DTACC','AUIN','AUOUT']]
            ndf = df.groupby(['LOSS','SCORE','OUT-DATA'], as_index=False)[['TNR','AUROC']].agg(['mean','std','count'])
            #############################################################################################################################
            #############################################################################################################################

            #"""
            print("RESULTS FOR EACH OOD SEPARATELY!!!")
            #############################################################################################################################
            #############################################################################################################################
            if data == 'cifar10':
                dfx = df.loc[df['OUT-DATA'].isin(['cifar100','imagenet_resize','lsun_resize','svhn'])]
            elif data == 'cifar100':
                dfx = df.loc[df['OUT-DATA'].isin(['cifar10','imagenet_resize','lsun_resize','svhn'])]
            elif data == 'imagenet1k':
                dfx = df.loc[df['OUT-DATA'].isin(['imagenet-o'])]
            #############################################################################################################################
            #############################################################################################################################
            ndf = dfx.groupby(['LOSS','SCORE','OUT-DATA']).agg(
                mean_AUROC=('AUROC', 'mean'), std_AUROC=('AUROC', 'std'), count_AUROC=('AUROC', 'count'),
                #mean_AUIN=('AUIN', 'mean'), std_AUIN=('AUIN', 'std'), count_AUIN=('AUIN', 'count'),
                mean_AUOUT=('AUOUT', 'mean'), std_AUOUT=('AUOUT', 'std'), count_AUOUT=('AUOUT', 'count'),
                mean_TNR=('TNR', 'mean'), std_TNR=('TNR', 'std'), count_TNR=('TNR', 'count'))
            nndf = ndf.sort_values(['LOSS','SCORE','OUT-DATA'], ascending=True)
            print(nndf)
            print()
            #"""

            #"""
            print("RESULTS FOR ALL OOD COMBINED!!!")
            #############################################################################################################################
            #############################################################################################################################
            if data == 'cifar10':
                dfx = df.loc[df['OUT-DATA'].isin(['cifar100','imagenet_resize','lsun_resize','svhn'])]
            elif data == 'cifar100':
                dfx = df.loc[df['OUT-DATA'].isin(['cifar10','imagenet_resize','lsun_resize','svhn'])]
            elif data == 'imagenet1k':
                dfx = df.loc[df['OUT-DATA'].isin(['imagenet-o'])]
            #############################################################################################################################
            #############################################################################################################################
            ndf = dfx.groupby(['LOSS','SCORE','OUT-DATA']).agg(
                mean_AUROC=('AUROC', 'mean'), std_AUROC=('AUROC', 'std'), count_AUROC=('AUROC', 'count'),
                mean_AUOUT=('AUOUT', 'mean'), std_AUOUT=('AUOUT', 'std'), count_AUOUT=('AUOUT', 'count'),
                mean_TNR=('TNR', 'mean'), std_TNR=('TNR', 'std'), count_TNR=('TNR', 'count'),)
            nndf = ndf.groupby(['LOSS','SCORE']).agg(
                mean_mean_AUROC=('mean_AUROC', 'mean'), mean_std_AUROC=('std_AUROC', 'mean'), count_mean_AUROC=('mean_AUROC', 'count'),
                mean_mean_AUOUT=('mean_AUOUT', 'mean'), mean_std_AUOUT=('std_AUOUT', 'mean'), count_mean_AUOUT=('mean_AUOUT', 'count'),
                mean_mean_TNR=('mean_TNR', 'mean'), mean_std_TNR=('std_TNR', 'mean'), count_mean_TNR=('mean_TNR', 'count'), )
            nndf = nndf.sort_values(['mean_mean_AUROC'], ascending=False)
            print(nndf)
            print()
            #"""


    print("\n################################################")
    print("################################################")
    print("################################################")
    print("###### TABLE: EXPECTED CALIBRATION ERROR #######")
    print("################################################")
    print("################################################")
    print("################################################\n")
    data_frame_list = []
    for file in file_names_dict_of_lists['results_calib.csv']:
        data_frame_list.append(pd.read_csv(file))
    best_results_data_frame = pd.concat(data_frame_list)
    best_results_data_frame.to_csv(os.path.join(path, 'all_results_calib.csv'), index=False)

    for data in DATASETS:
        for model in MODELS:
            print("\n########")
            print(data)
            print(model)
            df = best_results_data_frame.loc[
                best_results_data_frame['DATA'].isin([data]) &
                best_results_data_frame['MODEL'].isin([model])
            ]
            dft = df.groupby(['LOSS'], as_index=False)[['ECE']].agg(['mean','std','count'])
            dft = dft.sort_values([('ECE','mean')], ascending=True)#.drop_duplicates(["LOSS"])
            print(dft)
            print("########\n")


if __name__ == '__main__':
    main()
