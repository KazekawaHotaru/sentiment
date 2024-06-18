from torch import no_grad
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

epochs = 10
learning_rate = 1e-3
batch_size = 8
model_save_file = '/Users/yuyong/PycharmProjects/sentiment/model'
# # 设置为 MPS 设备，如果可用的话
# device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
# # 设置为 CUDA 设备，如果可用的话
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device = torch.device('mps')

# 从csv文件读取数据
data = pd.read_csv('data.csv')
validate = pd.read_csv('validate.csv')

train_texts = data['text'].tolist()
train_labels = data['label'].tolist()

validate_texts = validate['text'].tolist()
validate_labels = validate['label'].tolist()

# 加载预训练的 BERT 模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
model.to(device)

# 将文本转换成 token IDs，并进行填充
train_inputs = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True).to(device)
train_labels = torch.tensor(train_labels).to(device)

validate_inputs = tokenizer(validate_texts, return_tensors='pt', padding=True, truncation=True).to(device)
validate_labels = torch.tensor(validate_labels).to(device)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_inputs.input_ids, train_inputs.attention_mask, train_labels)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

validate_dataset = TensorDataset(validate_inputs.input_ids, validate_inputs.attention_mask, validate_labels)
validate_loader = DataLoader(validate_dataset, batch_size, shuffle=False)

# 设置 TensorBoardX 写入器
writer = SummaryWriter('runs')

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)

# 训练模型
model.train()
step = 0

train_step = 0
test_step = 0

total_test_step = 0
for epoch in tqdm(range(epochs)):
    print(f'Epoch {epoch + 1}/{epochs}')
    for i, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_step += 1
        print(f'Step {i}/{len(train_loader)}, Loss: {loss.item()}')
        # 记录训练损失
        writer.add_scalar("train_loss", loss.item(), train_step)

        if i + 1 % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item()})')

    # 进行评估
    model.eval()
    correct = 0
    total = 0
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(validate_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            total_test_loss = total_test_loss + loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            test_step += 1
            writer.add_scalar("test_loss", loss.item(), test_step)
            # total_accuracy = correct / total
        accuracy = correct / total
        print(f'准确率: {accuracy:.2f}')
        writer.add_scalar("test_accuracy", accuracy, epoch + 1)

    # 保存微调后的模型
    model.save_pretrained(f'{model_save_file}/sentiment{epoch}')
