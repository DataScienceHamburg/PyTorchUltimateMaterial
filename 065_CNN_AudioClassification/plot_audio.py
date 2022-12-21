import matplotlib.pyplot as plt
import torch

#%%
def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)

# source: adapted from https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
def plot_specgram(waveform, sample_rate, file_path = 'test2.png'):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    
    fig, axes = plt.subplots( num_channels, 1)
    fig.set_size_inches(10, 10)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    plt.gca().set_axis_off()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # plt.show(block=False)
    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0)
    