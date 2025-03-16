#这个文件的作用是：

# 读取 mbti_1.csv 数据
# 预处理文本数据
# 转换 MBTI 类型为标签
# 封装成 PyTorch Dataset 供训练使用

#运行方式
# cd D:\Workspace\AI_Workspace\MBTI_Transformer\src
# python dataset.py

import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# 读取数据集
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 预处理数据
class MBTIDataset(Dataset):
    def __init__(self, texts, labels):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.encodings = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

# 运行测试
if __name__ == "__main__":
    df = load_data("../data/mbti_1.csv")
    
    # 进行标签编码
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["type"])
    
    # 生成数据集
    dataset = MBTIDataset(df["posts"].tolist(), df["label"].values)
    print(f"数据集大小: {len(dataset)}")
