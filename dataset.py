import glob
import random
from torch.utils.data import Dataset

class CubeDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = sorted(glob.glob(f"{data_dir}/training*_formatted.txt"))
        self.index = []
        self.total_lines = 0
        for i, path in enumerate(self.file_paths):
            with open(path, 'r') as file:
                lines = file.readlines()
                self.total_lines += len(lines)
                self.index.extend([(i, j) for j in range(len(lines))])

    def __len__(self):
        return self.total_lines

    def __getitem__(self, idx):
        file_idx, line_idx = self.index[idx]
        target_file_path = self.file_paths[file_idx]

        with open(target_file_path, 'r') as file:
            lines = file.readlines()
            line = lines[line_idx].strip()