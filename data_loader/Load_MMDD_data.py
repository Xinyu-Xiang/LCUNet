import pickle
from torch.utils.data import Dataset
import numpy as np
import sys


class MMDD_test(Dataset):
    def __init__(self, list_dir, transform=None):
        """
        Same as training class, but loads 'test.pkl'.
        """
        # 临时修复 numpy._core 问题
        # if not hasattr(np, '_core'):
        #     sys.modules['numpy._core'] = np.core
        # with open('your_file.pickle', 'rb') as f:
        #     data = pickle.load(f)

        with open(list_dir, 'rb') as f:
            self.data_list = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Same output dict as train, but note that your stored
        test samples already have the extra leading dims.
        """
        sample = self.data_list[idx]
        video_name = sample['videoname']

        out = {
            'video_x': sample['video_x'],  # shape (1,32,224,224,3)
            'x_affect': sample['x_affect'],  # shape (1,64,7)
            'audio_x': sample['audio_x'],  # shape (1,224,224,3)
            'OpenFace_x': sample['OpenFace_x'],  # shape (1,64,43)
            'audio_wave': sample['audio_wave'],  # shape (1, T)
            'videoname': sample.get('videoname', None)
        }

        if self.transform:
            out = self.transform(out)

        return out, video_name
    


