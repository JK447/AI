# AI
あなたの専用AIを開発しましょう(Let's develop your dedicated AI.)
import os
import datetime
import torch
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer

FEEDBACK_FILE = 'feedbacks.txt'
AI_NAME_FILE = 'ai_name.txt'
API_KEY = "1c276a9b3c90b5ff58274df27db704df"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric&lang=ja"

# 日本語の県名から英語の都市名へのマッピング
CITY_MAPPING = {
    "東京": "Tokyo",
    "大阪": "Osaka",
    "福岡": "Fukuoka",
    "名古屋": "Nagoya",
    "札幌": "Sapporo",
    "広島": "Hiroshima",
    "京都": "Kyoto",
    "神戸": "Kobe",
    "横浜": "Yokohama",
    "仙台": "Sendai",
    "新潟": "Niigata"
}

def create_session():
    session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    return session_id

def handle_input(input_text):
    if input_text.startswith("http://") or input_text.startswith("https://"):
        return f"Received an image URL: {input_text}"
    else:
        return input_text

def education_mode(question):
    if "history" in question:
        return "History is the study of past events."
    elif "math" in question:
        return "Mathematics is the abstract study of topics such as quantity, structure, space, and change."
    else:
        return "Sorry, I don't have information on that topic."

def get_news():
    return "Here's today's top news: ..."

def get_weather(city="Tokyo"):
    # 日本語の都市名を英語の都市名に変換
    english_city = CITY_MAPPING.get(city, city)
    url = BASE_URL.format(english_city, API_KEY)
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"Sorry, I couldn't fetch the weather information for {city} right now."

    data = response.json()
    city_name = data['name']
    description = data['weather'][0]['description']
    temp = data['main']['temp']
    return f"Today's weather in {city_name} is {description} with a temperature of {temp}°C."

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

    current_session = create_session()

    while True:
        try:
            user_input = handle_input(input("You: "))

            if user_input.lower() == "exit":
                print(f"{ai_name}: さようなら！(Bye)")
                break
            elif "news" in user_input.lower():
                response = get_news()
            elif "天気" in user_input.lower():
                city = user_input.replace("天気", "").strip()
                if not city:
                    city = "Tokyo"
                response = get_weather(city)
            elif "teach me" in user_input.lower():
                response = education_mode(user_input)
            else:
                input_ids = tokenizer.encode(user_input, return_tensors="pt")
                with torch.no_grad():
                    output = model.generate(input_ids, max_length=150, num_return_sequences=1,
                                            no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id,
                                            temperature=0.8, top_k=50)
                response = tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"{ai_name}: {response}")

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
