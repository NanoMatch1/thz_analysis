import numpy as np
import pandas as pd

def interpolate_to_max_resolution(*args, **kwargs):
    '''Interpolate multiple datasets to the maximum resolution among them. Compatible with DataFrames, ndarrays, and dicts of 'data' and 'headers'.'''    
    object_list = [arg for arg in args]
    format_list = []
    new_data_list = []


    for i, obj in enumerate(object_list):
        if isinstance(obj, pd.DataFrame):
            data = obj.to_numpy()
            headers = obj.columns.tolist()
            format_list.append('dataframe')
        elif isinstance(obj, np.ndarray):
            data = obj
            headers = None
            format_list.append('ndarray')
        elif isinstance(obj, dict) and 'data' in obj and 'headers' in obj:
            data = obj['data']
            headers = obj['headers']
            format_list.append('dict')
        else:
            raise TypeError(f"Unsupported data type: {type(obj)}")
        
        new_data_list.append((data, headers))
    
    # Determine the maximum resolution based on the first column (assumed to be the x-axis)
    resolutions = []
    for data, headers in new_data_list:
        if len(data.shape) < 2:
            resolution = len(data)
        elif data.shape[1] < 2:
            resolution = len(data)
        else:
            resolution = len(data[:, 0])

        resolutions.append(resolution)

    # breakpoint()
    max_resolution = max(resolutions)
    new_data_list_resampled = []

    for data, headers in new_data_list:
        data.shape
        if len(data.shape) == 1:
            data = np.reshape(data, (len(data), 1)) # make it 2D for consistency    
        initial_axis = data[:, 0]
        new_freq_axis = np.linspace(initial_axis[0], initial_axis[-1], max_resolution)
        new_data = [new_freq_axis]

        for col in range(1, data.shape[1]):
            new_col = np.interp(new_freq_axis, initial_axis, data[:, col])
            new_data.append(new_col)

        resampled_data = np.column_stack(new_data)
        new_data_list_resampled.append((resampled_data, headers))

    # Convert back to original format
    final_results = []
    for i, (data, headers) in enumerate(new_data_list_resampled):
        if format_list[i] == 'dataframe':
            df = pd.DataFrame(data, columns=headers)
            final_results.append(df)
        elif format_list[i] == 'ndarray':
            final_results.append(data)
        elif format_list[i] == 'dict':
            final_results.append({'data': data, 'headers': headers})

    return final_results