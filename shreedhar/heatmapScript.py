import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_heatmap_from_csv(csv_file, output_folder):
    df = pd.read_csv(csv_file, header=0)

    plot_title = df.columns[0]
    row_labels = df.columns[1:].tolist()
    col_labels = df.iloc[:, 0].tolist()
    data = df.iloc[:, 1:].values

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(data, cmap='Reds', vmin=0, vmax=1)

    # Set row and column labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Display the values in each cell (optional)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = data[i, j]
            text_color = 'white' if value > 0.8 else 'black'
            text = ax.text(j, i, f'{value:.3f}', ha='center', va='center', color=text_color,
                           fontsize=10)

    # Create a colorbar
    # cbar = fig.colorbar(cax)

    plt.title(plot_title)

    # Save the heatmap as a PDF in the specified folder
    pdf_file = os.path.splitext(os.path.basename(csv_file))[0] + '.pdf'
    pdf_path = os.path.join(output_folder, pdf_file)
    plt.savefig(pdf_path, dpi=300)

    plt.tight_layout()
    plt.show()


def epExperiments():
    # Folder path for EP experiments
    data_folder = '/Users/shreedharjangam/ucsc/xai/codingExercise/visualGenerationCode/epExperiments'
    output_folder = '/Users/shreedharjangam/ucsc/xai/codingExercise/visualGenerationCode/epExperiments/epHeatmaps'

    # List of CSV files for EP experiments
    ep_csv_files = [
        'epFeatureAgreement.csv',
        'epRankAgreement.csv',
        'epSignAgreement.csv',
        'epSignedRankAgreement.csv',
        'epRankCorrelation.csv',
        'epPairwiseRankAgreement.csv',
        'epIOU.csv'
    ]

    # Iterate through EP CSV files and generate plots
    for csv_file in ep_csv_files:
        csv_file_path = os.path.join(data_folder, csv_file)
        plot_heatmap_from_csv(csv_file_path, output_folder)


def peExperiments():
    # Folder path for PE experiments
    data_folder = '/Users/shreedharjangam/ucsc/xai/codingExercise/visualGenerationCode/peExperiments'
    output_folder = '/Users/shreedharjangam/ucsc/xai/codingExercise/visualGenerationCode/peExperiments/peHeatmaps'

    # List of CSV files for PE experiments
    pe_csv_files = [
        'peFeatureAgreement.csv',
        'peRankAgreement.csv',
        'peSignAgreement.csv',
        'peSignedRankAgreement.csv',
        'peRankCorrelation.csv',
        'pePairwiseRankAgreement.csv',
        'peIOU.csv'
    ]

    # Iterate through PE CSV files and generate plots
    for csv_file in pe_csv_files:
        csv_file_path = os.path.join(data_folder, csv_file)
        plot_heatmap_from_csv(csv_file_path, output_folder)


def main():
    peExperiments()
    epExperiments()


if __name__ == "__main__":
    main()
