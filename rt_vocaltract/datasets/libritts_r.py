import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

class EtoELibriTTSRDataset(Dataset):
    def __init__(self, data_path, split="none"):

        self.sr = 24000

        # print("--- loading LibriTTS_R dataset ---")
        self.data_path = pathlib.Path(data_path)

        self.wav_root = self.data_path / "wavs"
        self.feat_root = self.data_path / "features"
        self.split_path = self.data_path / "split" / f"{split}.txt"
        self.spk_emb_root = pathlib.Path("/data/cheoljun/LibriTTS_R/spk_ft_wavlm")

        # print(f"-- wav root: {self.wav_root} ---")
        # print(f"-- feature root: {self.feat_root} ---")

        if split == "none":
            self.wavs = sorted(list(self.wav_root.glob("*.wav")))
            self.feats = sorted(list(self.feat_root.glob("*.npy")))
        else:
            self.wavs = []
            self.feats = []
            for line in open(self.split_path):
                line = str(line)
                self.wavs.append(self.wav_root / f"{line[:line.find('|')]}")
                self.feats.append(self.feat_root / f"{line[:line.find('|')].replace('wav', 'npy')}")

        # print("--- loaded LibriTTS_R dataset ---")

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        feat = self.feats[idx]

        wlm_feat = np.load(self.spk_emb_root / wav.name.replace("wav", "npy"))
        wlm_feat = torch.from_numpy(wlm_feat).float().unsqueeze(0)

        wav, _ = torchaudio.load(wav)

        feat_dict = np.load(feat, allow_pickle=True).item()

        ema = feat_dict["ema"]
        ema_len = ema.shape[0]

        pitch = feat_dict["pitch"][:ema_len].reshape(-1, 1)

        # Go from pitch range [50, 550] to [0, 5]
        pitch -= 50.0
        pitch /= 100

        feat = np.concatenate([ema, pitch], axis=1)

        feat = torch.from_numpy(feat).float()

        return wav, feat, wlm_feat

    @classmethod
    def collate(cls, col_len=0.1, ar=False, ar_len=512):
        def collate_fn(batch):
            wavs, feats, wlm_feats = zip(*batch)
            wavs = list(wavs)
            feats = list(feats)

            wav_col_len = int(col_len * 24000)
            feat_col_len = int(col_len * 50)
            # print(wav_col_len, feat_col_len)

            col_wavs = []
            col_feats = []
            col_wlm_feats = []

            for i in range(len(wavs)):
                # Pad to the nearest multiple of col_len (with leeway of 10 samples)
                wav_remain = wavs[i].shape[1] % wav_col_len
                wav_pad = wav_col_len - wav_remain if wav_remain > 10 or wav_col_len - wav_remain < 10 else 0
                wavs[i] = F.pad(wavs[i], (int(wav_pad), 0), value=0.0)

                # Truncate to the nearest multiple of col_len if higher
                wavs[i] = wavs[i][:, :int(wavs[i].shape[1] // wav_col_len) * wav_col_len]

                # Pad to the nearest multiple of col_len (with leeway of 3 frames)
                feats[i] = feats[i].transpose(1, 0)
                feat_remain = feats[i].shape[1] % feat_col_len
                feat_pad = feat_col_len - feat_remain if feat_remain > 3 or feat_col_len - feat_remain > 3 else 0
                feats[i] = F.pad(feats[i], (int(feat_pad), 0), value=0.0)

                # Truncate to the nearest multiple of col_len if higher
                feats[i] = feats[i][:, :int(feats[i].shape[1] // feat_col_len) * feat_col_len]

                col_wavs.extend(torch.chunk(wavs[i], int(wavs[i].shape[1] // wav_col_len), dim=1))
                col_feats.extend(torch.chunk(feats[i], int(feats[i].shape[1] // feat_col_len), dim=1))
                col_wlm_feats.extend([wlm_feats[i]] * int(wavs[i].shape[1] // wav_col_len))


            col_wavs = torch.stack(col_wavs)
            col_feats = torch.stack(col_feats)

            wlm_feats = torch.stack(col_wlm_feats)

            if ar:
                ars = [wav[:, -ar_len:] for wav in col_wavs[1:]]
                ars.insert(0, torch.zeros_like(ars[0]))
                ars = torch.stack(ars)
                return col_wavs, col_feats, ars, wlm_feats
            else:
                return col_wavs, col_feats

        return collate_fn

class LibriTTSRDataset(Dataset):
    MAX_WAV_LEN = 1052164
    MAX_FEAT_LEN = 2191

    def __init__(self, data_path, split="none", periodicity=True):

        # print("--- loading LibriTTS_R dataset ---")
        self.data_path = pathlib.Path(data_path)

        self.wav_root = self.data_path / "wavs"
        self.feat_root = self.data_path / "features"
        self.split_path = self.data_path / "split" / f"{split}.txt"

        self.periodicity = periodicity

        # print(f"-- wav root: {self.wav_root} ---")
        # print(f"-- feature root: {self.feat_root} ---")

        if split == "none":
            self.wavs = sorted(list(self.wav_root.glob("*.wav")))
            self.feats = sorted(list(self.feat_root.glob("*.npy")))
        else:
            self.wavs = []
            self.feats = []
            for line in open(self.split_path):
                line = str(line)
                self.wavs.append(self.wav_root / f"{line[:line.find('|')]}")
                self.feats.append(self.feat_root / f"{line[:line.find('|')].replace('wav', 'npy')}")

        # print("--- loaded LibriTTS_R dataset ---")

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        feat = self.feats[idx]

        wav, _ = torchaudio.load(wav)

        feat_dict = np.load(feat, allow_pickle=True).item()

        ema = feat_dict["ema"]
        ema_len = ema.shape[0]
        periodicity = feat_dict["periodicity"][:ema_len].reshape(-1, 1)

        pitch = feat_dict["pitch"][:ema_len].reshape(-1, 1)
        pitch -= 5.0
        pitch /= 450.0

        if self.periodicity:
            feat = np.concatenate([ema, periodicity, pitch], axis=1)
        else:
            feat = np.concatenate([ema, pitch], axis=1) # for old model
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
    dataset = LibriTTSRDataset(data_path, split="train")
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    wav, feat = next(iter(dataloader))
    print(wav.shape, feat.shape)
