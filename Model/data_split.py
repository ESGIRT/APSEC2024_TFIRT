import pandas as pd

def split_dataset(csv_file_path, train_ratio=0.6, val_ratio=0.2, random_state=42):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Group by category
    grouped = df.groupby('category', group_keys=False)

    # Shuffle the order of each group
    df_shuffled = grouped.apply(lambda x: x.sample(frac=1, random_state=random_state, replace=True))

    # Calculate the number of samples for each category
    category_counts = df_shuffled['category'].value_counts()

    train_category_counts = (category_counts * train_ratio).astype(int)
    val_category_counts = (category_counts * val_ratio).astype(int)
    test_category_counts = category_counts - train_category_counts - val_category_counts

    # Split the dataset
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for category in category_counts.index:
        category_data = df_shuffled[df_shuffled['category'] == category]

        train_samples = category_data[:train_category_counts[category]]
        val_samples = category_data[
                      train_category_counts[category]:train_category_counts[category] + val_category_counts[category]]
        test_samples = category_data[train_category_counts[category] + val_category_counts[category]:]

        train_data = pd.concat([train_data, train_samples], ignore_index=True)
        val_data = pd.concat([val_data, val_samples], ignore_index=True)
        test_data = pd.concat([test_data, test_samples], ignore_index=True)

    test_data.to_csv("test_data.csv", index=False)

    return train_data, val_data, test_data
