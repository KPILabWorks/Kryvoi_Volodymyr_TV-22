import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': range(1, 1_000_001),
    'text': np.random.choice([
        "Думи мої, думи мої,",
        "Лихо мені з вами!",
        "Нащо стали на папері",
        "Сумними рядами?..",
        "Чом вас вітер не розвіяв",
        "В степу, як пилину?",
        "Чом вас лихо не приспало,",
        "Як свою дитину?..",
        "Бо вас лихо на світ на сміх породило,",
        "Поливали сльози... Чом не затопили,",
        "Не винесли в море, не розмили в полі?",
        "Не питали б люди, що в мене болить,",
        "Не питали б, за що проклинаю долю,",
        "Чого нужу світом? «Нічого робить», —",
        "Не сказали б на сміх..."
    ], size=1_000_000)
})

def info_metric(text):
    vowels = 'аеєиіїоуюяaeiou'
    vowel_count = sum(1 for ch in text.lower() if ch in vowels)
    word_count = len(text.split())
    length = len(text)
    return round((length * vowel_count) / (word_count + 1), 2)  # +1 для уникнення ділення на 0

df['info_metric'] = df['text'].apply(info_metric)

print(df.head())

