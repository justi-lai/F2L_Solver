import torch
from torch.utils.data import IterableDataset, get_worker_info
import os
import glob
import random

class CubeDataset(IterableDataset):
    """
    Streaming Dataset for sharded .pt files.
    Loads chunks sequentially and shuffles within chunks.
    """
    def __init__(self, data_path_or_files, max_len=128):
        """
        Args:
            data_path_or_files: Directory path (str) OR list of file paths (List[str]).
            max_len (int): Max sequence length.
        """
        self.max_len = max_len
        
        if isinstance(data_path_or_files, list):
            self.files = data_path_or_files
        elif isinstance(data_path_or_files, str) and os.path.isdir(data_path_or_files):
            self.files = sorted(glob.glob(os.path.join(data_path_or_files, "part_*.pt")))
        else:
             raise ValueError("data_path_or_files must be a dir or list of files")
             
        if not self.files:
            raise ValueError(f"No part_*.pt files found.")
            
        print(f"Dataset initialized with {len(self.files)} shards.")
        
        # Vocab size fallback
        self.vocab_size = 38 

    def __iter__(self):
        worker_info = get_worker_info()
        
        # Split files among workers
        if worker_info is None:
            my_files = self.files
        else:
            # Simple splitting: worker 0 gets files [0, N, 2N...], worker 1 gets [1, N+1...]?
            # Or interleaved?
            # Interleaved is simpler:
            my_files = [f for i, f in enumerate(self.files) if i % worker_info.num_workers == worker_info.id]
            
        # Shuffle file order (chunk shuffling)
        # We perform shallow copy to not affect other epochs if persistent? 
        # Actually __iter__ is called new each time.
        random.shuffle(my_files)
        
        for file_path in my_files:
            data = torch.load(file_path)
            
            # Shuffle items within chunk
            random.shuffle(data)
            
            for item in data:
                yield self._process_item(item)

    def _process_item(self, item):
        states = item["states"] 
        actions = item["actions"]
        
        L = states.shape[0]
        if L > self.max_len:
            states = states[:self.max_len]
            actions = actions[:self.max_len]
            L = self.max_len
            
        padded_states = torch.zeros(self.max_len, 20, 24)
        padded_states[:L] = states
        
        padded_actions = torch.zeros(self.max_len, dtype=torch.long)
        padded_actions[:L] = actions
        
        mask = torch.zeros(self.max_len, dtype=torch.long)
        mask[:L] = 1
        
        return padded_states, padded_actions, mask
        
    def __len__(self):
        # Optional: Estimate length (num files * chunk size?)
        # Useful for tqdm if consistent.
        # Let's assume 10k per chunk if standard.
        # Just return rough estimate (len(files) * 10000).
        return len(self.files) * 10000
