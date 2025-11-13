import torch
import sys
import os
import argparse
# Добавляем импорт AutoTokenizer из библиотеки HuggingFace transformers
from transformers import AutoTokenizer

from previous_chapters import GPTModel

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step: (B, T, C) becomes (B, C)
        logits = logits[:, -1, :] 

        # Apply softmax to get probabilities (Optional for greedy search)
        # probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get the token with the highest probability (greedy choice)
        _, idx_next = torch.topk(logits, k=1, dim=-1) # (B, 1)

        # Append predicted next token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# Функции text_to_token_ids и token_ids_to_text адаптированы для токенизатора HF
def text_to_token_ids(text, tokenizer):
    # Метод encode у токенизатора HF возвращает список int
    # Мы не указываем allowed_special, так как HF токенизатор обрабатывает это иначе
    encoded = tokenizer.encode(text) 
    # Добавляем размерность батча: [num_tokens] -> [1, num_tokens]
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # Удаляем размерность батча: [1, num_tokens] -> [num_tokens]
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# --- Основная часть скрипта ---

# Setup Argument Parser
parser = argparse.ArgumentParser(description="GPT2 Text Generation Script.")
parser.add_argument("-p", "--prompt", type=str, default="Who is Jack Gisburn", 
                    help="The initial prompt for text generation. Defaults to 'Who is Jack Gisburn'.")
parser.add_argument("-n", "--max_tokens", type=int, default=50, 
                    help="Maximum number of new tokens to generate. Defaults to 50.")
parser.add_argument("-m", "--model_path", type=str, default='model.pth',
                    help="Path to the model .pth file. Defaults to 'model.pth'.")
args = parser.parse_args()

# 1. Загрузка токенизатора (Используем HF вместо tiktoken)

# Имя токенизатора должно соответствовать тому, который вы использовали при ОБУЧЕНИИ
# Например, bert-base-multilingual-cased или DeepPavlov/rubert-base-cased
TOKENIZER_NAME = "bert-base-multilingual-cased" 

try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Tokenizer '{TOKENIZER_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer {TOKENIZER_NAME}: {e}")
    sys.exit(1)

# 2. Определение параметров модели (Configuration dictionary)
# ВАЖНО: vocab_size должен соответствовать размеру словаря вашего НОВОГО токенизатора!
# Размер словаря для bert-base-multilingual-cased равен 119547
GPT_CONFIG = {
    "vocab_size": tokenizer.vocab_size, # Используем размер словаря из загруженного токенизатора
    "context_length": 256,  # Максимальная длина контекста (должна совпадать с обучением)
    "emb_dim": 768,         # Размерность эмбеддингов (должна совпадать с обучением)
    "n_heads": 12,          # Количество голов внимания (должна совпадать с обучением)
    "n_layers": 12,         # Количество трансформерных блоков (должна совпадать с обучением)
    "drop_rate": 0.1,       # Процент Dropout
    "qkv_bias": False       # Использовать ли bias в QKV-линейных слоях
}

# 3. Инициализация модели
model = GPTModel(GPT_CONFIG)

# 4. Загрузка весов из вашего файла .pth
PTH_FILE_PATH = 'model.pth' 

if os.path.exists(PTH_FILE_PATH):
    # Загружаем state_dict в модель
    # map_location='cpu' гарантирует, что она загрузится даже без GPU
    model.load_state_dict(torch.load(PTH_FILE_PATH, map_location=torch.device('cpu'), weights_only=True))
    print(f"Model successfully loaded from {PTH_FILE_PATH}")
else:
    print(f"Error: Model file not found at path {PTH_FILE_PATH}")
    sys.exit(1)

# Переводим модель в режим оценки (важно для Dropout/BatchNorm)
model.eval() 

# 5. Generate text using the provided prompt and max tokens
start_prompt = args.prompt
max_new_tokens = args.max_tokens

print(f"\nStarting Prompt: '{start_prompt}'")
print(f"Max new tokens: {max_new_tokens}")


# Преобразование текста в токены
input_ids = text_to_token_ids(start_prompt, tokenizer)

# Запуск генерации
generated_ids = generate_text_simple(
    model=model, 
    idx=input_ids, 
    max_new_tokens=max_new_tokens, 
    context_size=GPT_CONFIG["context_length"]
)

# Преобразование токенов обратно в текст
output_text = token_ids_to_text(generated_ids, tokenizer)

print("\n--- Generated Text ---")
print(output_text)
print("----------------------")
