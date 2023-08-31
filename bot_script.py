import telebot
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

import re

def extract_response(text, start_marker="@@ВТОРОЙ@@", end_marker="@@ПЕРВЫЙ@@"):
    text = re.sub(r'<pad>', '', text)
    
    # Пытаемся извлечь текст между @@ВТОРОЙ@@ и @@ПЕРВЫЙ@@
    match = re.search(f"{re.escape(start_marker)}(.+?){re.escape(end_marker)}", text)
    if match:
        return match.group(1).strip()
    
    # Если не удалось, извлекаем текст после @@ВТОРОЙ@@
    match = re.search(f"{re.escape(start_marker)}(.+)", text)
    if match:
        return match.group(1).strip()
    
    return "ОШИБКА"


# Инициализация бота
bot = telebot.TeleBot("6403748773:AAHkJeF4ZL0gPn3UgjLCx6wF2CJM_a7DNoo")

# Загрузка предобученной модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained("fine-tuned ccm2")

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "Привет!")

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_message = message.text
    inputs = tokenizer('@@ПЕРВЫЙ@@ ' + user_message + ' @@ВТОРОЙ@@ ', return_tensors='pt')
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=5,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.3,
        repetition_penalty=1.2,
        length_penalty=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=40
    )
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
    context_with_response_cleaned = [extract_response(text) for text in context_with_response]
    context_with_response_cleaned2 = [text for text in context_with_response]
    bot.send_message(message.chat.id, ''.join(context_with_response_cleaned))
    #bot.send_message(message.chat.id, ''.join(context_with_response_cleaned2))
    print(context_with_response_cleaned2)

# Запуск бота
bot.polling()
