import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-root", type=str, default="G:/Depression/EDaci_Woz/data")
    parser.add_argument("--modality-id", type=str, default="audio_mfcc")
    parser.add_argument("--dest-root", type=str, default="G:/Depression/EDaci_Woz/no-chunked")
    args = parser.parse_args()

    feature_dir = "features"
    featureID = "OpenSMILE2.3.0_mfcc"

    dest_dir = os.path.join(args.dest_root, args.modality_id + "_vggish")
    os.makedirs(dest_dir, exist_ok = True)

    # 加载预训练的wav2vec 2.0模型
    model = Wav2Vec2Model.from_pretrained("G:/pretrained_models/wav2vec2-base-960h")

    sessionIDs = sorted( os.listdir(args.src_root) )
    for sessionID in tqdm(sessionIDs):
        feature_path = os.path.join(args.src_root, sessionID, feature_dir, sessionID.split("_")[0]+"_"+featureID+".csv")
        # feature_path = os.path.join(args.src_root, sessionID, feature_dir, sessionID+"_"+featureID+".csv")
        df = pd.read_csv(feature_path, delimiter=";")

        # (64848, 39) （片段个数，特征数）
        seq = df.iloc[:, 2:].to_numpy()

        with torch.no_grad():
            outputs = model(seq)

        # 提取最后一层的隐藏状态作为特征表示
        last_hidden_states = outputs.last_hidden_state

        dest_path = os.path.join(dest_dir, sessionID+".npz")
        np.savez_compressed(dest_path, data=seq)
