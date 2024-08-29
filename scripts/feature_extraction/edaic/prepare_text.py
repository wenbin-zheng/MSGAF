import os
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('G:/pretrained_models/bert-base-uncased')

if __name__ == '__main__':
    data_dir = r'G:/Depression/EDaci_Woz/data/'


    # 获取文件夹
    sessionIDs = sorted(os.listdir(data_dir))
    # 获取文稿
    for sessionID in tqdm(sessionIDs):

        data = pd.read_csv(os.path.join(data_dir, sessionID, sessionID.split('_')[0] + '_Transcript.csv'))
        data = data['Text']
        # 合并所有文本
        concat_data = ("".join(i for i in data))
        # 使用bert进行词嵌入
        encode_bert = bert_tokenizer.encode_plus(concat_data, add_special_tokens=True, pad_to_max_length=True)

        print()


    print()