from thz.dataset import DataSet

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
    data_set = DataSet(file_dir=file_dir, sample_keys=['sample'], reference_keys=['reference'])
    data_set.load_all_data()
    datadict = data_set.data_dict
    # test = data_set.grabone()
    # test.plot_current()
    # test.plot_current()
    # test._interpolate_time_axis(new_limits=(, 140))
    # data_set.interpolate_pulse_window()
    data_set.prepare_all_for_fft(length_factor=5, dc_points=10)


    data_set.plot_current()
    data_set.plot_sn()
    print(datadict)
    breakpoint()  # For debugging purposes