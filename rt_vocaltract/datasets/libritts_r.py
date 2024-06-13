import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class LibriTTSRDataset(Dataset):
    MAX_WAV_LEN = 1052164
    MAX_FEAT_LEN = 2191

    def __init__(self, data_path):

        print("--- loading LibriTTS_R dataset ---")
        self.data_path = pathlib.Path(data_path)

        self.wav_root = self.data_path / "wavs"
        self.feat_root = self.data_path / "features"

        print(f"-- wav root: {self.wav_root} ---")
        print(f"-- feature root: {self.feat_root} ---")

        self.wavs = sorted(list(self.wav_root.glob("*.wav")))
        self.feats = sorted(list(self.feat_root.glob("*.npy")))

        print("--- loaded LibriTTS_R dataset ---")

    def __len__(self):
        return len(list(self.wav_root.glob("*.wav")))

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        feat = self.feats[idx]

        wav, _ = torchaudio.load(wav)

        feat_dict = np.load(feat, allow_pickle=True).item()

        ema = feat_dict["ema"]
        ema_len = ema.shape[0]
        periodicity = feat_dict["periodicity"][:ema_len].reshape(-1, 1)
        pitch = feat_dict["pitch"][:ema_len].reshape(-1, 1)

        feat = np.concatenate([ema, periodicity, pitch], axis=1)
        feat = torch.from_numpy(feat).float()

        wav_pad = torch.zeros(1, self.MAX_WAV_LEN - wav.shape[1])
        wav = torch.cat([wav_pad, wav], dim=1)

        feat = LibriTTSRDataset.pad_feat(feat)

        return wav, feat
    
    @classmethod
    def pad_feat(cls, feat, device=None):
        if len(feat.shape) == 2:
            feat_pad = torch.zeros(cls.MAX_FEAT_LEN - feat.shape[0], feat.shape[1])
            return torch.cat([feat_pad, feat], dim=0)
        else:
            feat_pad = torch.zeros(feat.shape[0], cls.MAX_FEAT_LEN - feat.shape[1], feat.shape[2])
            feat_pad = feat_pad.to(device)
            return torch.cat([feat_pad, feat], dim=1)


if __name__ == "__main__":
    data_path = "/data/common/LibriTTS_R"
    dataset = LibriTTSRDataset(data_path)
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    wav, feat = next(iter(dataloader))
    print(wav.shape, feat.shape)
