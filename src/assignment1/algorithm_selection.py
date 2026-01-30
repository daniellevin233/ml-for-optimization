"""
Algorithm Selection Framework for Social Golfer Problem.

Uses Random Forest classifier to predict the best algorithm (SAT vs SA)
for a given problem instance based on instance features.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.utils import find_project_root


class AlgorithmSelector:
    """
    Algorithm selection model using Random Forest.

    Predicts whether SAT or Simulated Annealing is better for a given instance.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize the algorithm selector.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

        self.feature_columns = ['m', 'n', 'w', 'total_golfers', 'problem_size', 'sa_runtime', 'sat_runtime', 'sat_result','sa_result','sa_repeated_pairs']
        self.target_column = 'best_algo'

        self.is_trained = False
        self.train_accuracy = None
        self.test_accuracy = None

    def load_data(
        self,
        data_path: Optional[Path] = None,
        test_size: float = 0.35,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split the combined dataset.

        Args:
            data_path: Path to combined_dataset.csv
            test_size: Proportion of data for testing (default 0.35 = 35%)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if data_path is None:
            data_path = find_project_root() / "data" / "combined_dataset.csv"

        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        print(f"Total instances: {len(df)}")

        # Extract features and target
        X = df[self.feature_columns]
        y = df[self.target_column]

        # Split into train/test with shuffling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=y,  # Maintain class distribution
        )

        print(f"\nTrain set: {len(X_train)} instances ({len(X_train)/len(df)*100:.1f}%)")
        print(f"Test set: {len(X_test)} instances ({len(X_test)/len(df)*100:.1f}%)")
        print(f"\nTrain distribution: {y_train.value_counts().to_dict()}")
        print(f"Test distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """
        Train the Random Forest classifier.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*60)

        print(f"Features: {self.feature_columns}")
        print(f"Number of estimators: {self.n_estimators}")
        print(f"Max depth: {self.max_depth if self.max_depth else 'Unlimited'}")

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate training accuracy
        y_train_pred = self.model.predict(X_train)
        self.train_accuracy = accuracy_score(y_train, y_train_pred)

        print(f"\nTraining complete!")
        print(f"Training accuracy: {self.train_accuracy:.4f}")

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        print("\n" + "="*60)
        print("EVALUATING MODEL ON TEST SET")
        print("="*60)

        y_pred = self.model.predict(X_test)
        self.test_accuracy = accuracy_score(y_test, y_pred)

        print(f"\nTest accuracy: {self.test_accuracy:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['SA', 'SAT']))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=['sa', 'sat'])
        print(f"              Predicted SA  Predicted SAT")
        print(f"Actual SA     {cm[0,0]:<13} {cm[0,1]:<13}")
        print(f"Actual SAT    {cm[1,0]:<13} {cm[1,1]:<13}")

        return {
            'test_accuracy': self.test_accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from the trained model.

        Returns:
            DataFrame with features and their importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances,
        }).sort_values('importance', ascending=False)

        return feature_importance_df

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict best algorithm for given instances.

        Args:
            X: Features (must have same columns as training)

        Returns:
            Array of predictions ('sat' or 'sa')
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        return self.model.predict(X)


def run_algorithm_selection_experiment(
    test_size: float = 0.35,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> AlgorithmSelector:
    """
    Run complete algorithm selection experiment.

    Milestones:
    1. Load and split data (65/35 train/test)
    2. Train Random Forest classifier
    3. Evaluate on test set
    4. Analyze feature importance

    Args:
        test_size: Proportion for test set (0.35 = 35%)
        n_estimators: Number of trees
        max_depth: Max tree depth

    Returns:
        Trained AlgorithmSelector
    """
    print("="*60)
    print("ALGORITHM SELECTION EXPERIMENT")
    print("="*60)

    # Milestone 1: Load and split data
    print("\n[Milestone 1] Loading and splitting data...")
    selector = AlgorithmSelector(
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    X_train, X_test, y_train, y_test = selector.load_data(test_size=test_size)

    # Milestone 2: Train model
    print("\n[Milestone 2] Training Random Forest...")
    selector.train(X_train, y_train)

    # Milestone 3: Evaluate
    print("\n[Milestone 3] Evaluating on test set...")
    metrics = selector.evaluate(X_test, y_test)
    print("\nEvaluation Metrics:")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("Classification Report:")
    print(pd.DataFrame(metrics['classification_report']).transpose())

    # Milestone 4: Feature importance
    print("\n[Milestone 4] Analyzing feature importance...")
    feature_importance = selector.get_feature_importance()
    print("\nFeature Importances:")
    print(feature_importance.to_string(index=False))

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Train Accuracy: {selector.train_accuracy:.4f}")
    print(f"Test Accuracy: {selector.test_accuracy:.4f}")

    return selector


if __name__ == "__main__":
    selector = run_algorithm_selection_experiment(
        test_size=0.35,
        n_estimators=100,
        max_depth=None,
    )