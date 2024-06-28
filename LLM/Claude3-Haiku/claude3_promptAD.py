import pandas as pd
import openai
import csv
import re
import os
from sklearn.metrics import classification_report

openai.api_base = ""
openai.api_key = ""

df = pd.read_csv('test_data.csv')

file_labels = {
    0: 'describe',
    1: 'expected',
    2: 'reproduce',
    3: 'actual',
    4: 'environment',
    5: 'additional'
}

file_content = {
    0: 'describe: this category mainly guides to provide a brief description of the bug',
    1: 'expected: this category mainly guides to describe what would happen without the bug',
    2: 'reproduce: this category mainly guides to describe the steps to reproduce the bug',
    3: 'actual: this category mainly guides to describe what would happen with the bug',
    4: 'environment: this category mainly guides to describe where the bug happened',
    5: 'additional: this category mainly guides to describe any other context about the bug'
}

def classify_text(text):
    prompt = f"Classify the following text into one of these categories: {list(file_labels.values())}, " \
             f"To help you make a better judgment, I'll start by explaining to you what each category means: {list(file_content.values())}\n" \
             f"Text: {text}\n" \
             f"just output to determine what kind of text this is"

    completion =openai.ChatCompletion.create(
        model="claude-3-haiku",
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

        with open('claude3_promptAD.csv', 'a', newline='', encoding='utf-8') as file:
            fieldnames = ['predictions', 'true_labels']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({
                'predictions': predictions,
                'true_labels': true_labels
            })

        print(f"Processed {i} entries, Model prediction: {category}, Correct result: {label}")
