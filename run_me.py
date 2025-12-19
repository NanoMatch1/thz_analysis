import os
from thz.dataset import DataSet, Constants


file_dir = os.path.join(os.path.dirname(__file__), 'thz', 'data')

if __name__ == "__main__":
    import numpy as np

    constants = Constants(
        thickness = 4e-4,
        eps_inf = 1, # glass 3.42, Si 11.6
        ns = 1.9, #substate n
)
    file_dir = r'C:\Users\Samuel\Data\THz\noisetest'
    # file_dir = r'C:\Users\Samuel\Data\THz\Sam\13-11-25_Co-HHTP'

    data_set = DataSet(file_dir=file_dir, sample_keys=['sample'], reference_keys=['reference'])
    data_set.load_all_data()
    data_set.load_constants(constants)
    # data_set.plot_current()
    data_set.calculate_std_dev_all(show_graph=True, limit=10, normalise=True)
    data_set.calculate_SNR_all(show_graph=True, limit=10)
    
    # test = data_set.grabone()
    # test.plot_current()
    # test.plot_current()
    # test._interpolate_time_axis(new_limits=(, 140))
    # data_set.interpolate_pulse_window()
    data_set.group_files()
    # data_set.center_pad_window_all(length_factor=5, window_alpha=0.2)
    # data_set.centerpad_legacy()
    # data_set.edge_window_pad()
    data_set.centerpad_legacy()
    data_set.prepare_for_fft_all(pad_length_factor=1, window_alpha=0.6, baseline_points=10)
    # data_set.plot_current()
    data_set.fft_set()
    # data_set.plot_fft_current(series='fft_raw')
    # data_set.plot_fft_current(series='fft_edge_windowed')
    # data_set.plot_fft_current(series='fft_centered_padded')
    transfer, phase = data_set.fft_compare()




    # TODO List:
    # - decide on best centering/padding method and remove others
    # - clean data type handling in dataset (DataFrame vs numpy array vs dict)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2, figsize=(12,8))
    # plt.show()

    # breakpoint()


    for idx, (series, data_dict) in enumerate(transfer.items()):
            # order the dict based on temperature
        data_dict = dict(sorted(data_dict.items(), key=lambda item: float(item[0].split('_')[2].split('K')[0])))
        
        colormap = plt.get_cmap('plasma')
        colors = colormap(np.linspace(0, 1, len(data_dict)))

        for index, (filename, output) in enumerate(data_dict.items()):
            data = output['transfer']
            calculations = output['physical parameters']
            real_c = calculations['Real Conductivity ($\\sigma_{1}$)']
            imag_c = calculations['Imaginary Conductivity ($\\sigma_{2}$)']

            print(f"Series: {series}, File: {filename}")

            data_x = data['Frequency (THz)'][1:]  # BUG: need to find source of array mismatch, I think it is earlier when dividing by zero in conductivity calc
            data_y = data['Amplitude'][1:]


            ax[0].plot(data_x, data_y, label='{}:{}'.format(filename, 'Amplitude'), color=colors[index])
            ax[1].plot(data_x, real_c, label='{}:{}'.format(filename, 'Real Conductivity'), color=colors[index])
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