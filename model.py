import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import optim
import load_data
from keras.datasets import imdb
from keras.preprocessing import sequence
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
# X_train = sequence.pad_sequences(X_train, maxlen=500)

# Step 1: Reconstruction Model
# -----------------------------

# Huggingface Transformers taking the T5 base model.
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# the following hyper parameter are task-specific
max_source_length = 512
max_target_length = 512

# get the datasets
imdb_train, _ = load_data.imdb_dataset(train=True)
imdb_test, _ = load_data.imdb_dataset(test=True)
# random split of data
train_percent = int(0.8 * len(imdb_train))
dev_percent = int(0.2 * len(imdb_train))
imdb_dev = imdb_train[train_percent:]
imdb_train = imdb_train[:train_percent]
dev_len = len(imdb_dev)
task_prefix = "translate English to English: "
for entries in imdb_train:
    input_sequence = entries['text'].split(".")
    output_sequence = input_sequence
    encoding = tokenizer(
        [task_prefix + sequence for sequence in input_sequence],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    target_encoding = tokenizer(
        output_sequence,
        padding="longest",
        max_length=max_target_length,
        truncation=True
    )
    labels = target_encoding.input_ids

    labels = torch.tensor(labels)
    labels[labels == tokenizer.pad_token_id] = -100
    loss = model(input_ids=input_ids,
                 attention_mask=attention_mask,
                 labels=labels).loss
    loss.backward()
# Step 2: Style Discriminator:  4-layer fully connect neural networks.
# --------------------------------------------------------------------

# Hyper-parameters
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 50
# inner_learning_rate
alpha = 0.0001
#outer_learning_rate
beta = 0.001

# Use leave one out evaluation method
# For each iteration, two source domains are randomly selected as meta-training domain. Rest: meta-validation domain

# Evaluation:
# 1. Human Metric
# 2. BLEU
# 3. Style Control (S-Acc) - Style classifier pretrained on the dataset
# 4. Domain Control (D-Acc) - verifies whether generated sentences have the characteristics of the target domain with
#                              a pretrained domain classifier





