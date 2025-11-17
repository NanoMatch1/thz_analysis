from thz.dataset import DataSet

file_dir = r'C:\Users\Samuel\Data\THz\Sam\13-11-25_Co-HHTP'

if __name__ == "__main__":
    data_set = DataSet(file_dir=file_dir, sample_keys=['sample'], reference_keys=['reference'])