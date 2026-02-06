import os
from thz.dataset import DataSet, Constants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

file_dir = os.path.join(os.path.dirname(__file__), 'thz', 'data')

def compare_noise_jan(scale='linear'):
    fileDir = r'C:\Users\Samuel\Data\THz\noisetest\2026-02-03\comparison'

    data_set = DataSet(file_dir=fileDir, sample_keys=['sample'], reference_keys=['reference'])
    data_set.load_all_data()

    # for filename, thzdata in data_set.data.items():
    #     average = []
    #     data = thzdata.raw_data[20:30, 1:]
    #     for idx, line in enumerate(data):
    #         std_dev = np.std(line, axis=0)
    #         average.append(std_dev)
    #         print(f"File: {filename}, Line {idx+20}, Baseline Std Dev: {std_dev}")
    #     print(np.average(average))
    #     std_dev = np.std(data, axis=0)
    #     print(f"File: {filename}, Baseline Std Dev: {std_dev}")

    # fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True)
    fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig)
    plt.subplots_adjust(hspace=0.3, wspace=0.1)


    # Row-spanning title axes
    title_ax0 = fig.add_subplot(gs[0, :])
    title_ax1 = fig.add_subplot(gs[1, :])
    title_ax2 = fig.add_subplot(gs[2, :])

    for ta in (title_ax0, title_ax1, title_ax2):
        ta.axis("off")

    ax0 = fig.add_subplot(gs[0, 0])
    ax5 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    ax = [ax0, ax1, ax2, ax3, ax4, ax5]

    for filename, thzdata in data_set.data.items():
        print(thzdata.filename)
        print(thzdata)
        filename = filename[:-4]
        SNR_data = thzdata.calculate_SNR(limit=None)

        x_axis = np.arange(len(thzdata.data[:, 0]))
        raw_SNR = SNR_data['peak_amplitude'] / SNR_data['baseline_std']
        std = SNR_data['std']
        baseline = np.average(thzdata.data[:10, 1])
        initial_scan = thzdata.raw_data[:20, 1] 
        # initial_scan = initial_scan - np.min(initial_scan)
        # thzdata.data[:, 1] = thzdata.data[:, 1] -  min(thzdata.data[:, 1])
        thzdata.data[:, 1] = thzdata.data[:, 1] - baseline
        ax[0].plot(x_axis, thzdata.data[:, 1]*1000, label='{}\n raw total SNR: {:.2f}'.format(filename, raw_SNR))
        ax[5].plot(np.arange(len(initial_scan)), initial_scan*1000, label='Initial Scan: {}'.format(filename))
        ax[1].plot(x_axis, SNR_data['snr'], label=filename)
        ax[3].plot(x_axis, SNR_data['snr'], label=filename)
        ax[2].plot(x_axis, std, label=filename)
        ax[4].plot(x_axis, std, label=filename)

    ax[0].set_title('Time Traces')
    ax[5].set_title('First 20 points of single scan')
    # ax[1].set_title('Signal-to-Noise Ratio Per time point (SNR) (Higher = better)')
    # ax[2].set_title('Standard Deviation of Signal (V) (Lower = Better)', loc='right')
    # title_ax0.set_title("Time-domain traces", fontsize=12, pad=10)
    title_ax1.set_title("Signal-to-noise ratio per time point (SNR) (HIGHER = better)", fontsize=12, pad=10)
    title_ax2.set_title("Standard Deviation of signal (V) (LOWER = better)", fontsize=12, pad=10)

    ax[0].set_ylabel('Signal Amplitude (mV)')
    ax[1].set_ylabel('SNR')
    ax[2].set_ylabel('Standard Deviation (V)')
    ax[2].set_xlabel('Time Point Index')
    ax[1].grid(True, which='major', linewidth=1)
    ax[1].grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)
    ax[2].grid(True, which='major', linewidth=1)
    ax[2].grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)
    ax[3].grid(True, which='major', linewidth=1)
    ax[3].grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)
    ax[4].grid(True, which='major', linewidth=1)
    ax[4].grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.5)
    ax[0].legend()
    # ax[1].legend()
    # ax[2].legend()
    # if scale == 'log':
        # ax[0].set_yscale('log')
    # ax[5].set_yscale('log')
    ax[3].set_yscale('log')
    ax[4].set_yscale('log')
    plt.show()

def monitor_analysis(data_set: DataSet):
    # monotor analysis function
    data_set.load_all_files()

        # manual x axis for comparison -------
    for filename, thzdata in data_set.data.items():
        # thzdata.raw_data = thzdata.raw_data[:30, :]
        x_axis = np.arange(thzdata.raw_data.shape[0])
        new_data = np.column_stack((x_axis, thzdata.raw_data[:,1], np.zeros_like(x_axis)))
        thzdata._time_data = new_data
        thzdata._data = new_data

    data_set.plot_current(line_alpha=0.5)

    for filename, thz_data in data_set.data.items():
        data = thz_data.data[:, 1]
        # plt.plot(data, label=filename)
        # plt.legend()
        # plt.show()
        std_dev = np.std(data, axis=0)
        # std_dev = round(std_dev, 10)
        print(f"File: {filename}, Std Dev: {std_dev}")

    breakpoint()

if __name__ == "__main__":
    import numpy as np

    constants = Constants(
        thickness = 4e-4,
        eps_inf = 1, # glass 3.42, Si 11.6
        ns = 1.9, #substate n
)
    file_dir = r'C:\Users\Samuel\Data\THz\Sam\13-11-25_Co-HHTP'
    file_dir = r'C:\Users\Samuel\Data\THz\noisetest\2026-01-26'
    file_dir = r'C:\Users\Samuel\Data\THz\noisetest\test_15_QWP'
    file_dir = r'C:\Users\Samuel\Data\THz\noisetest\test_16'
    # file_dir = r'C:\Users\Samuel\Data\THz\Dani\2026-01-29_germanium'
    # file_dir = r'C:\Users\Samuel\Data\THz\noisetest\2026-01-29_hero-scan'
    file_dir = r'C:\Users\Samuel\Data\THz\noisetest\2026-02-03\comparison'
    file_dir = r'C:\Users\Samuel\Data\Chris'
    # file_dir = r'C:\Users\Samuel\Data\THz\Sam\2026-02-06_MINTS'
    # r'C:\Users\Samuel\Data\THz\noisetest\test_13_comparison'

    # compare_noise_jan() # Compare the noise levels before/after modifications
    # compare_noise_jan(scale='log') # Compare the noise levels before/after modifications

    data_set = DataSet(file_dir=file_dir, sample_keys=['sample'], reference_keys=['reference'])
    data_set.load_all_data()
    print(data_set.data)
    
    # data_set.plot_current()

    # monitor_analysis(data_set)


    data_set.data.info
    data_set.group_files(keywords=['type', 'series'])

    # references = data_set.data.references
    # samples = data_set.data.samples

    # data_set.plot_all_reference_and_data()
    # fig, ax = plt.subplots(2, 1, figsize=(10,8), sharex=True)

    # # for filename, fileitem in references.items():
    # #     thzdata = data_set.data[filename]
    # #     ax[0].plot(thzdata.data[:,0], thzdata.data[:,1], label=filename)
    # # for filename, fileitem in samples.items():
    # #     thzdata = data_set.data[filename]
    # #     ax[1].plot(thzdata.data[:,0], thzdata.data[:,1], label=filename)
    # for filename, thzdata in data_set.data.items():
    #     if data_set.data.is_reference(filename):
    #         ax[0].plot(thzdata.data[:,0], thzdata.data[:,1], label=filename)
    #     elif data_set.data.is_sample(filename):
    #         ax[1].plot(thzdata.data[:,0], thzdata.data[:,1], label=filename)
        
    
    # ax[0].set_title('References')
    # ax[0].legend()
    # ax[1].set_title('Samples')
    # ax[1].legend()
    # plt.show()
    data_set.align_on_peak()
    data_set.plot_current(index_axis=True)


    # data_set.plot_current()
    # data_set.centerpad_legacy()
    data_set.prepare_for_fft_all(pad_length_factor=2.5, window_alpha=0.6, baseline_points=10, show_graph=True)
    data_set.fft_set()
    # data_set.plot_fft_current(series='fft_raw')
    # data_set.plot_fft_current(series='fft_edge_windowed')
    # data_set.plot_fft_current(series='fft_centered_padded')
    transfer, phase = data_set.fft_compare()
    # breakpoint()




    # TODO List:
    # - decide on best centering/padding method and remove others
    # - clean data type handling in dataset (DataFrame vs numpy array vs dict)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2, figsize=(12,8))
    # plt.show()

    # breakpoint()


    for idx, (series, data_dict) in enumerate(transfer.items()):
        #     # order the dict based on temperature
        # data_dict = dict(sorted(data_dict.items(), key=lambda item: float(item[0].split('_')[2].split('K')[0])))
        
        colormap = plt.get_cmap('viridis')
        colors = colormap(np.linspace(0, 1, len(data_dict)))

        for index, (filename, output) in enumerate(data_dict.items()):
            data = output['transfer']
            calculations = output['physical parameters']
            real_c = calculations['Real Conductivity ($\\sigma_{1}$)']
            imag_c = calculations['Imaginary Conductivity ($\\sigma_{2}$)']

            print(f"Series: {series}, File: {filename}")

            data_x = data['Frequency (THz)'][1:]  # BUG: need to find source of array mismatch, I think it is earlier when dividing by zero in conductivity calc
            data_y = data['Amplitude'][1:]
            #clamp data from 0.2 to 2.7 thz
            skipindex = np.where(data_x > 0.2)[0][0]
            endindex = np.where(data_x > 2.7)[0][0]


            ax[0].plot(data_x[skipindex:endindex], data_y[skipindex:endindex], label='{}:{}'.format(filename, 'Amplitude'), color=colors[index])
            ax[1].plot(data_x[skipindex:endindex], real_c[skipindex:endindex], label='{}:{}'.format(filename, 'Real Conductivity'), color=colors[index])
            # ax[2].plot(data_x, imag_c, label='{}:{}'.format(filename, 'Imaginary Conductivity'), color=colors[index])



        ax[0].legend()
        ax[0].set_title(f'Transfer Function: {series}')
        ax[0].set_xlabel('Frequency (THz)')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_xlim(0, 3)

        ax[1].set_xlabel('Frequency (THz)')
        ax[1].set_ylabel('Real Conductivity ($\\sigma_{1}$)')
        ax[1].set_xlim(0, 3)
        # ax[1].set_ylim())
        ax[1].legend()

        # ax[2].set_xlabel('Frequency (THz)')
        # ax[2].set_ylabel('Imaginary Conductivity ($\\sigma_{2}$)')
        # ax[2].set_xlim(0, 3)
        # ax[2].legend()

        # ax[0].set_ylim())

    plt.show()
            # plt.plot(data['Frequency (THz)'], data['Amplitude'], label='Amplitude', label='{}:{}')
    # sn = data_set.compare_snr()

    # LEgacy marker


    # for filename, thzdata in datadict.items():
    #     if 'sample' in filename.lower():
    #         test = data_set.grouper(filename)
    #         breakpoint()
    # data = data_set.grabone()
    # datadict = data_set.compute_all_constants(constants)
    breakpoint()
    data_set.plot_current()
    # data_set.plot_sn()
    # print(datadict)
    breakpoint()  # For debugging purposes