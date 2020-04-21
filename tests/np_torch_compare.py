import argparse
import os

import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pystoi import stoi
from torch_stoi import NegSTOILoss

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', type=str, default=None,
                    help='Text file with all wav files to test. We assume'
                         'that for each test file s1.wav, s1_est.wav exists.')


def main(file_list):
    labels = ['np', 'np_ext', 'pt_vad_ext', 'pt_vad', 'pt_ext', 'pt', 'fs', 'fname']
    files_df = pd.read_csv(file_list)
    stoi_df = pd.DataFrame(columns=labels)
    for i, wav_file in tqdm(enumerate(files_df['filename']), total=len(files_df)):
        # Read files
        # if i != 538:
        #     continue
        # else:
        #     import ipdb; ipdb.set_trace()
        try:
            clean, fs = sf.read(wav_file, dtype='float32')
            enh, fs = sf.read(wav_file.replace('.wav', '_est.wav'), dtype='float32')
        except:
            print("Could not read file ", wav_file)
            continue
        line = make_line(clean, enh, fs)
        line += [fs, wav_file]
        stoi_df.loc[len(stoi_df)] = line

        mix_name = os.path.join('/'.join(wav_file.split('/')[:-1]), 'mix.wav')
        try:
            enh, fs = sf.read(mix_name, dtype='float32')
        except:
            print("Could not read file ", mix_name)
            continue
        line = make_line(clean, enh, fs)
        line += [fs, wav_file + '.mix']
        stoi_df.loc[len(stoi_df)] = line

        if i % 250 == 0 and i > 0:
            df_name = 'partial{}_df.csv'.format(2*i)
            stoi_df.to_csv(df_name)
    return stoi_df


def make_line(clean, enh, fs):
    # Compute in NumPy
    line = [stoi(clean, enh, fs), stoi(clean, enh, fs, extended=True)]
    # Compute in PyTorch
    for use_vad in [True, False]:
        for extended in [True, False]:
            loss = NegSTOILoss(sample_rate=fs, use_vad=use_vad,
                               extended=extended)
            line.append(-loss(torch.from_numpy(enh),
                              torch.from_numpy(clean)).item())
    return line


def plot_fromdf(df):

    return


def do_plot(x, y, data, title=None):
    u = np.arange(1000) / 1000  # To make diagonal
    fig = plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x, y=y, data=data)
    plt.xlim(0, 1)
    plt.ylim(0, 1.5)
    plt.xlabel('NumPy (pystoi)')
    plt.ylabel('PyTorch (here)')
    plt.plot(u, u)
    plt.title(title)
    filename = title.replace(' ', '').replace(',', '').replace('/', '')
    plt.savefig('../plots/{}.png'.format(filename), dpi=200)
    plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    # df = main(args.file_list)
    df = pd.read_csv('partial56000_df.csv')
    # plot_fromdf(df)
    u = np.arange(1000) / 1000  # To make diagonal

    # 8K plots
    df8k = df[df['fs'] == 8000.0]
    df8k = df8k[df8k['np'] != 1e-5]  # Very short utts

    do_plot('np', 'pt_vad', df8k, title="8kHz with VAD")
    do_plot('np', 'pt', df8k, title="8kHz w/o VAD")
    do_plot('np_ext', 'pt_vad_ext', df8k, title="8kHz Extended with VAD")
    do_plot('np_ext', 'pt_ext', df8k, title="8kHz Extended w/o VAD")

    # 16k plots
    df16k = df[df['fs'] == 16000.0]
    do_plot('np', 'pt_vad', df16k, title="16kHz with VAD")
    do_plot('np', 'pt', df16k, title="16kHz w/o VAD")
    do_plot('np_ext', 'pt_vad_ext', df16k, title="16kHz Extended with VAD")
    do_plot('np_ext', 'pt_ext', df16k, title="16kHz Extended w/o VAD")
