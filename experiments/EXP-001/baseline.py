"""Small deterministic baseline using scikit-learn's built-in digits dataset."""

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:
    data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(f"accuracy={accuracy_score(y_test, predictions):.6f}")


if __name__ == "__main__":
    main()
