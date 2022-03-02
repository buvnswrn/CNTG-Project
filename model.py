from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import optim
# Huggingface Transformers taking the T5 base model.
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Style Discriminator:  4-layer fully connect neural networks.

# Hyper-parameters
optimizer = optim.Adam(model.parameters(), lr=1e-5)
epochs = 50
inner_learning_rate_alpha = 0.0001
outer_learning_rate_beta = 0.001

# Use leave one out evaluation method
# For each iteration, two source domains are randomly selected as meta-training domain. Rest: meta-validation domain

# Evaluation:
# 1. Human Metric
# 2. BLEU
# 3. Style Control (S-Acc) - Style classifier pretrained on the dataset
# 4. Domain Control (D-Acc) - verifies whether generated sentences have the characteristics of the target domain with
#                              a pretrained domain classifier





