import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import argparse

# === 命令列參數 ===
parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, default=6, help="HuBERT layer index to extract")
args = parser.parse_args()

# === 路徑設定 ===
metadata_path = "/scratch/s5944562/HuBERT/utterance_table_mix_segmented_split.csv"
output_dir = f"/scratch/s5944562/HuBERT/features_mix_layer{args.layer}"
os.makedirs(output_dir, exist_ok=True)

# === 載入模型與 feature extractor ===
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True).eval()
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

def extract_features(df_split, split_name):
    X_list, y_list = [], []
    
    X_path = os.path.join(output_dir, f"X_{split_name}.npy")
    y_path = os.path.join(output_dir, f"y_{split_name}.npy")

    # === Resume 支援 ===
    processed_files = set()
    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"🔁 Resuming {split_name} from saved features...")
        X_list = list(np.load(X_path))
        y_list = list(np.load(y_path))
        processed_files = set(df_split.iloc[:len(X_list)]["file_path"])  # 已處理的檔案

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Extracting {split_name}"):
        path = row["file_path"]
        label = row["label"]

        if path in processed_files:
            continue

        try:
            waveform, sr = torchaudio.load(path)

            # Downmix stereo
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

            # Feature extraction
            inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
                layer_feat = out.hidden_states[args.layer]  # Flexible layer
                pooled = layer_feat.mean(dim=1).squeeze().numpy()

            if pooled.shape == (768,):
                X_list.append(pooled)
                y_list.append(label)
            else:
                print(f"⚠️ Skipped {path} due to invalid shape {pooled.shape}")

        except Exception as e:
            print(f"❌ Failed on {path}: {e}")

    # 儲存
    np.save(X_path, np.array(X_list))
    np.save(y_path, np.array(y_list))
    print(f"✅ Saved {split_name}: {len(X_list)} samples")

# === 主程式 ===
df = pd.read_csv(metadata_path)
for split in ["train", "val", "test"]:
    df_split = df[df["split"] == split]
    extract_features(df_split, split)