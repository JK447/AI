import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

FEEDBACK_FILE = 'feedbacks.txt'
AI_NAME_FILE = 'ai_name.txt'

def save_feedback(input_text, model_response, feedback):
    with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{input_text}\t{model_response}\t{feedback}\n")

def save_ai_name(ai_name):
    with open(AI_NAME_FILE, 'w', encoding='utf-8') as f:
        f.write(ai_name)

def get_ai_name():
    if os.path.exists(AI_NAME_FILE):
        with open(AI_NAME_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def chat_with_gpt2():
    ai_name = get_ai_name()
    if ai_name is None:
        ai_name = input("AIの名前を入力してください(Please enter a name.): ")
        save_ai_name(ai_name)
    
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    print(f"{ai_name}: こんにちは！どうかしましたか？(Hello! Is there something you need?)")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print(f"{ai_name}: さようなら！(By)")
                break

            input_ids = tokenizer.encode(user_input, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(input_ids, max_length=150, num_return_sequences=1, 
                                        no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id, 
                                        temperature=0.8, top_k=50)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"{ai_name}: {response}")

            # フィードバックの自動記録
            feedback = "auto"
            save_feedback(user_input, response, feedback)

        except Exception as e:
            print(f"{ai_name}: 申し訳ございません、エラーが発生しました。")
            print(f"Error: {e}")

if __name__ == "__main__":
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            pass

    chat_with_gpt2()
