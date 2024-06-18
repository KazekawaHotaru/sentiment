from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 使用mps加速模型预测
device = torch.device('mps')
# 预测文本

# ====好评====
# text = ("这篇文章让我眼前一亮，作者的思考深度和广度都令人钦佩。"
#         "每一个观点都经过精心论证，逻辑清晰，条理分明。"
#         "语言流畅而富有感染力，让人在阅读中不禁产生共鸣。我期待着作者未来更多的作品。")

# text = ("这篇文章真是让我受益匪浅。作者不仅提供了丰富的信息和独到的见解，还以深入浅出的方式让我对某个领域有了全新的认识。"
#         "读完之后，我感受到了作者对于写作的热爱和专注，也激发了我对学习的热情。")

# ====差评====
# text = ("这篇文章让我感到有些失望。虽然作者试图表达某些观点，但整体来说，内容显得平淡无奇，缺乏新意。"
#         "文章的结构也比较松散，没有明确的主题和中心思想，让人在阅读中感到迷茫。希望作者能够加强思考和创作，提升文章的质量。")

text = ("这篇文章在语言表达上存在一些问题。一些句子结构复杂，用词生僻，让人难以理解。"
        "另外，文章中还存在一些逻辑上的漏洞和矛盾之处，让人对作者的观点产生怀疑。"
        "我认为作者需要认真审视自己的文章，进行必要地修改和完善。")

# 加载模型和 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 使用原始BERT模型
# model = BertForSequenceClassification.from_pretrained('bert-base-chinese').to(device)
# 使用微调后的模型
model = BertForSequenceClassification.from_pretrained('model').to(device)

# 将文本转换成 token IDs
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)

# 将输入数据输入到模型中进行预测
outputs = model(**inputs)
# 获取模型的预测结果
predictions = torch.softmax(outputs.logits, dim=1).tolist()[0]

if predictions[0] > predictions[1]:
    print("这是一个负面评价！")
else:
    print("这是一个正面评价！")

# 打印预测结果
print("情感概率:", predictions)
