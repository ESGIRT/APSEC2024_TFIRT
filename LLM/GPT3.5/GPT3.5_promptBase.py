import pandas as pd
from openai import OpenAI
import csv
import re

from sklearn.metrics import classification_report

client = OpenAI(api_key="Input your API key")

df = pd.read_csv('test_data.csv')

file_labels = {
    0: 'describe',
    1: 'expected',
    2: 'reproduce',
    3: 'actual',
    4: 'environment',
    5: 'additional'
}

def classify_text(text):
    prompt = f"Classify the following text into one of these categories: {list(file_labels.values())}\n" \
             f"Text: {text}\n" \
             f"just output to determine what kind of text this is"

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = completion.choices[0].message.content
    pattern = re.compile(r'\b(describe|expected|reproduce|actual|environment|additional)\b', re.IGNORECASE)

    category = set(match.lower() for match in pattern.findall(text))
    category = list(category)[0]

    for key, value in file_labels.items():
        if value == category:
            return key
    return -1

if __name__ == '__main__':

    predictions = []
    true_labels = []

    start_index = int(input("From which entry do you want to start crawling data? (Enter 1 to start from the first entry): ")) - 1

    for i, text in enumerate(df['element'][start_index:], start=start_index + 1):
        label = df['label'][i - 1]

        category = classify_text(text)
        predictions.append(category)
        true_labels.append(int(label))

        with open('GPT3.5_promptBase.csv', 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['predictions', 'true_labels']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({
                'predictions': predictions,
                'true_labels': true_labels
            })

        print(f"Processed {i} entries")

