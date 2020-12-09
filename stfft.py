import csv

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import xlsxwriter
from pandas import DataFrame


def main():
    file = "12.wav" #   ----->the wanted  audio  file in .wav format this song is 30 seconds song

    # load audio files with librosa
    y, sr = librosa.load(file, sr=511)
    # select frame_size and Hop_size
    FRAME_SIZE = 1024
    HOP_SIZE = 512
    # extract the stft
    S_scale = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    # calculate the spectrum of sound (z=x+iy)
    Y_scale_m = np.abs(S_scale) ** 2
    # transpose the matrix

    Y_scale = Y_scale_m[85:255]
    Y_scale_transpose = np.transpose(Y_scale)
    print(Y_scale.shape)
    # find the max frequency and max frequency index per frame
    Max = (np.max(np.array(Y_scale), axis=0))
    max_Index = Y_scale.argmax(axis=0)
    print(Max)
    print(max_Index)

    # export to csv
    workbook = xlsxwriter.Workbook('C:/Users/sarina/Desktop/stft.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(Y_scale_transpose):
        worksheet.write_column(row, col, data)
    workbook.close()
    # save result data to csv
    df= DataFrame({'max_frequency': max_Index, 'max_frequency_value': Max})
    df.to_excel('C:/Users/sarina/Desktop/results.xlsx', sheet_name='sheet1', index=False) #----> change to the wanted directory
    with open('C:/Users/sarina/Desktop/resultsofstft.csv', 'a', newline='') as csvfile: #----> change to the wanted directory
        filednames = ['max_frequency', 'max_frequency_value']
        thewriter = csv.DictWriter(csvfile, fieldnames=filednames)
        thewriter.writerow(
            {'max_frequency': max_Index, 'max_frequency_value': Max})

    def plot_spectrogram(Y, sr, hop_length, y_axis='linear'):
        plt.figure(figsize=(30, 10))
        librosa.display.specshow(Y, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
        plt.colorbar(format="%+2.f")

    plot_spectrogram(Y_scale, sr, HOP_SIZE)
    plt.show()


if __name__ == "__main__":
    main()
