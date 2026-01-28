import os
from thz.dataset import DataSet, Constants
import numpy as np
import matplotlib.pyplot as plt

file_dir = os.path.join(os.path.dirname(__file__), 'thz', 'data')

def monitor_analysis(data_set: DataSet):
    # monotor analysis function
    data_set.load_all_dat_files()

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
    # r'C:\Users\Samuel\Data\THz\noisetest\test_13_comparison'

    data_set = DataSet(file_dir=file_dir, sample_keys=['sample'], reference_keys=['reference'])

    # monitor_analysis(data_set)

    data_set.load_all_data()
    data_set.load_constants(constants)

    data_set.data.info
    data_set.group_files(keywords=['type', 'series', 'temp'])

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


    # data_set.plot_current()
    breakpoint()
    data_set.centerpad_legacy()
    data_set.prepare_for_fft_all(pad_length_factor=2.5, window_alpha=0.6, baseline_points=10, show_graph=False)
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