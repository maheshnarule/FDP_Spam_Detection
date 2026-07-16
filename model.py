from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def main():
    # Load the real spam dataset
    data_path = Path(__file__).with_name("spam.csv")
    df = pd.read_csv(data_path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "text"]

    # Convert labels to binary format: ham=0, spam=1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["text"]
    y = df["label"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert text to numeric features
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Support Vector Machine": SVC(kernel="linear", class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }

    print("\nModel Comparison on Spam Dataset\n" + "-" * 55)
    for name, model in classifiers.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{name}")
        print("Accuracy:", f"{acc:.4f}")
        print("Confusion Matrix:")
        print("                 Predicted")
        print("                Ham  Spam")
        print(f"Actual Ham     {cm[0, 0]:4d}  {cm[0, 1]:4d}")
        print(f"Actual Spam    {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    while True:
        print("\nChoose a model to test on sample emails:")
        for i, name in enumerate(classifiers.keys(), start=1):
            print(f"{i}. {name}")
        print("0. Exit")

        try:
            choice = int(input("Enter your choice: ").strip())
        except ValueError:
            print("Please enter a valid number.")
            continue

        if choice == 0:
            print("Exiting...")
            break

        if choice < 1 or choice > len(classifiers):
            print("Invalid choice. Try again.")
            continue

        selected_name = list(classifiers.keys())[choice - 1]
        selected_model = classifiers[selected_name]
        selected_model.fit(X_train_vec, y_train)

        sample_emails = [
            "Congratulations! You have won a free prize.",
            "Hi, are we meeting tomorrow?",
        ]
        sample_vec = vectorizer.transform(sample_emails)
        sample_pred = selected_model.predict(sample_vec)

        print(f"\nPredictions using {selected_name}:")
        for email, pred in zip(sample_emails, sample_pred):
            label = "Spam" if pred == 1 else "Ham"
            print(f"- {email}\n  Prediction: {label}")


if __name__ == "__main__":
    main()
