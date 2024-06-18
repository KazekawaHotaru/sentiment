import torch
import pandas as pd
from tqdm import tqdm
from torch import no_grad
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# 定义批处理大小
batch_size = 8
# 定义运行设备
device = torch.device('mps')

# 加载预训练的 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# 使用BERT模型
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese').to(device)
# 使用微调后的模型
model = BertForSequenceClassification.from_pretrained('model').to(device)

# 将输入数据输入到模型中进行预测
validate = pd.read_csv('validate.csv')

# 读取csv文件中的两列
validate_texts = validate['text'].tolist()
validate_labels = validate['label'].tolist()

# 将文本转换成 token IDs
validate_inputs = tokenizer(validate_texts, return_tensors='pt', padding=True, truncation=True).to(device)
validate_labels = torch.tensor(validate_labels).to(device)

# 加载数据集并创建DataLoader
validate_dataset = TensorDataset(validate_inputs.input_ids, validate_inputs.attention_mask, validate_labels)
validate_loader = DataLoader(validate_dataset, batch_size, shuffle=False)

# 开始模型评估
model.eval()

# 记录正确预测数和总数
correct = 0
total = batch_size * len(validate_loader)

# 使用no_grad关闭梯度计算
with no_grad():

    for input_ids, attention_mask, labels in tqdm(validate_loader):
        # 使用mps加速
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 获取输出
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 获取预测结果
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()

    print(f'正确预测数: {correct}')
    print(f'预测总数: {total}')
    accuracy = correct / total
    print(f'准确率: {accuracy:.2f}%')
