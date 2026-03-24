import matplotlib.pyplot as plt
import numpy as np

def inspect_phase(dataset, **kwargs) -> dict:
    data_dict = {}
    title = kwargs.get("title", "Phase Spectrum")
    fig, ax = plt.subplots(2, 1)
    for filename, data in dataset.data.items():
        # print(data.processing_dict.keys())
        spectrum = data.data
        ax[0].plot(spectrum[:, 0], spectrum[:, 1], label=filename)
        ax[1].plot(spectrum[:, 0], spectrum[:, 2], label=filename)
        ax[1].plot(spectrum[:, 0], np.unwrap(spectrum[:, 2]), label=filename+" unwrapped", linestyle="--")

        data_dict[filename] = {
            "frequency": spectrum[:, 0],
            "amplitude": spectrum[:, 1],
            "phase": spectrum[:, 2],
            "unwrapped_phase": np.unwrap(spectrum[:, 2])
        }

    ax[0].set_title("Amplitude Spectrum")
    ax[1].set_title(title)
    ax[0].legend()
    ax[1].legend()
    plt.show()
    
    return data_dict


