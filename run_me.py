from thz.dataset import DataSet, Constants

file_dir = r'C:\Users\Samuel\Data\THz\Sam\13-11-25_Co-HHTP'

def convert_to_pandas(thzdata):
    import pandas as pd

    data = thzdata.data
    time = data[:, 0]
    amplitude = data[:, 1]
    std_error = data[:, 2]

    df = pd.DataFrame(amplitude, columns=['Mean'])
    df.insert(0, 'Time (ps)', time)
    df['std error'] = std_error

    return df




if __name__ == "__main__":

    constants = Constants(
        thickness = 2e-4,
        eps_inf = 1, # glass 3.42, Si 11.6
        ns = 1.9, #substate n
)

    data_set = DataSet(file_dir=file_dir, sample_keys=['sample'], reference_keys=['reference'])
    data_set.load_all_data()
    datadict = data_set.data_dict
    # test = data_set.grabone()
    # test.plot_current()
    # test.plot_current()
    # test._interpolate_time_axis(new_limits=(, 140))
    # data_set.interpolate_pulse_window()
    data_set.group_files()
    data_set.prepare_all_for_fft(length_factor=5, dc_points=10)

    # LEgacy marker



    for filename, thzdata in datadict.items():
        if 'sample' in filename.lower():
            test = data_set.grouper(filename)
            breakpoint()


    data_set.plot_current()
    data_set.plot_sn()
    print(datadict)
    breakpoint()  # For debugging purposes