import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.gridspec import GridSpec


def plot_heatmap_from_csv(csv_file, output_folder, which):
    df = pd.read_csv(csv_file, header=0)

    plot_title = df.columns[0]
    row_labels = df.columns[1:].tolist()
    col_labels = df.iloc[:, 0].tolist()
    data = df.iloc[:, 1:].values
    if data.shape == (3, 3):
        order = [1, 2, 0]
        row_labels = [row_labels[o] for o in order]
        col_labels = [col_labels[o] for o in order]
        data = data[order, :]
        data = data[:, order]
    elif data.shape == (4, 4):
        order = [1, 2, 0, 3]
        row_labels = [row_labels[o] for o in order]
        col_labels = [col_labels[o] for o in order]
        data = data[order, :]
        data = data[:, order]


    plt.imshow(data, cmap='Reds', vmin=0, vmax=1)
    ax = plt.gca()

    # Set row and column labels
    tick_dict = {'EP': 'SelfExp', 'EP+Occ': 'Occl', 'EP+LIME': 'LIME', 
                 'PE': 'SelfExp', 'PE+Occ': 'Occl', 'PE+LIME': 'LIME', 
                 'Occlusion': 'Occl', 'SelfExp': 'SelfExp', 'LIME': 'LIME', 
                 'SelfExpTokK': 'TopK', 'SelfExpTopK': 'TopK', 'PE+Nat': 'TopK', 
                 'EP+Nat': 'TopK'}
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels([tick_dict.get(l.strip(), '!'+l.strip()) for l in col_labels], ha='center', fontsize=8)
    ax.set_yticklabels([tick_dict.get(l.strip(), '!'+l.strip()) for l in row_labels], rotation=90, va='center', fontsize=8)

    # Display the values in each cell (optional)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = data[i, j]
            text_color = 'white' if value > 0.8 else 'black'
            text = ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color,
                           fontsize=10)

    title_dict = {'PE Pairwise Rank Agreement': 'Pairwise\nRank Agreement', 
                  'PE Rank Correlation': 'Rank\nCorrelation', 
                  'PE Rank Agreement': 'Rank\nAgreement', 
                  'PE Sign Agreement': 'Sign\nAgreement', 
                  'PE Signed Rank Agreement': 'Signed\nRank Agreement', 
                  'EP Pairwise Rank Agreement': 'Pairwise\nRank Agreement', 
                  'EP Rank Correlation': 'Rank\nCorrelation', 
                  'EP Rank Agreement': 'Rank\nAgreement', 
                  'EP Sign Agreement': 'Sign\nAgreement', 
                  'EP Signed Rank Agreement': 'Signed\nRank Agreement', 
                  'EP Feature Agreement': 'E-P Feature Agreement', 
                  'PE Feature Agreement': 'P-E Feature Agreement', 
                  'EP IOU': 'E-P IoU', 
                  'PE IOU': 'P-E IoU'}

    print(plot_title)
    plt.title(f'{which} {title_dict.get(plot_title, plot_title)}', fontsize=10)

    # plt.show()


def epExperiments(gs):
    # Folder path for EP experiments
    data_folder = 'epExperiments'
    output_folder = 'ep_heatmaps'

    # List of CSV files for EP experiments
    ep_csv_files = [
        # 'epIOU.csv',
        # 'epFeatureAgreement.csv',
        'epPairwiseRankAgreement.csv',
        'epRankCorrelation.csv',
        'epRankAgreement.csv',
        'epSignAgreement.csv',
        'epSignedRankAgreement.csv'
    ]

    # Iterate through EP CSV files and generate plots
    for i, csv_file in enumerate(ep_csv_files):
        # plt.subplot(2, 5, i+1)
        plt.gcf().add_subplot(gs[0:4, i*4:(i+1)*4])
        csv_file_path = os.path.join(data_folder, csv_file)
        plot_heatmap_from_csv(csv_file_path, output_folder, 'E-P')

def peExperiments(gs):
    # Folder path for PE experiments
    data_folder = 'peExperiments'
    output_folder = 'pe_heatmaps'

    # List of CSV files for PE experiments
    pe_csv_files = [
        # 'peFeatureAgreement.csv',
        # 'peIOU.csv', 
        'pePairwiseRankAgreement.csv',
        'peRankCorrelation.csv',
        'peRankAgreement.csv',
        'peSignAgreement.csv',
        'peSignedRankAgreement.csv',
    ]

    # Iterate through PE CSV files and generate plots

    for i, csv_file in enumerate(pe_csv_files):
        # plt.subplot(2, 5, i+6)
        plt.gcf().add_subplot(gs[5:9, i*4:(i+1)*4])
        csv_file_path = os.path.join(data_folder, csv_file)
        plot_heatmap_from_csv(csv_file_path, output_folder, 'P-E')

def others(gs):
    files = ['epExperiments/epFeatureAgreement.csv', 'epExperiments/epIOU.csv', 
             'peExperiments/peFeatureAgreement.csv', 'peExperiments/peIOU.csv']
    for i, csv_file_path in enumerate(files):
        plt.gcf().add_subplot(gs[10:15, i*5:(i+1)*5])
        plot_heatmap_from_csv(csv_file_path, None, '')

def main():
    fig = plt.figure(figsize=[13, 15])
    gs = GridSpec(15, 20, figure=fig, wspace=0.1, hspace=1, height_ratios=[1]*4+[0.8]+[1]*4+[0]+[1]*5)
    epExperiments(gs)
    peExperiments(gs)
    others(gs)
    # plt.tight_layout()
    plt.savefig('disagreement2.pdf', dpi=fig.dpi)
    plt.show()

if __name__ == "__main__":
    main()
