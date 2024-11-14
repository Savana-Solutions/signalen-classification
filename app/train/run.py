import argparse
import logging

import joblib
import nltk

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from app.train.model_validation import (
    calculate_metrics,
    generate_and_save_confusion_matrix,
    save_detailed_validation_results,
)
from app.train.pre_processing import (
    filter_labels,
    generate_training_testing_data,
    load_and_clean_data,
    sample_and_create_label,
)
from app.train.train_model import (
    generate_category_paths,
    get_parameters,
    get_pipeline,
    perform_grid_search_cv,
)

# Create the parser and add arguments
parser = argparse.ArgumentParser(description="Script for training a model.")
parser.add_argument("--filepath",
                    type=str,
                    required=True,
                    help="Path to the input CSV file")
parser.add_argument("--columns",
                    type=lambda s: [item.lower() for item in s.split(",")],
                    required=True,
                    help="Columns to consider")
parser.add_argument("--frac",
                    type=float,
                    default=1.0,
                    help="Fraction for sampling")
parser.add_argument("--split",
                    type=float,
                    default=0.9,
                    help="Train/Test split ratio")
parser.add_argument("--output_dir",
                    type=str,
                    default="/app/output",
                    help="Output directory")
parser.add_argument('--save_detailed_validation',
                    type=bool,
                    nargs="?",
                    const=True,
                    default=False)


def balance_dataset(X, y, method='undersample'):
    if method == 'undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif method == 'oversample':
        sampler = RandomOverSampler(random_state=42)
    else:
        raise ValueError("Method must be 'undersample' or 'oversample'")
    
    X_resampled, y_resampled = sampler.fit_resample(X.to_frame(), y)
    return X_resampled.iloc[:, 0], y_resampled

if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Script started.")
    logging.info('-' * 80)

    # Log the parsed arguments
    logging.info("Arguments received: ")
    for arg, value in vars(args).items():
        logging.info(f" {arg}: {value}")

    # Downloading stopwords
    nltk.download("stopwords", raise_on_error=True)
    stop_words = nltk.corpus.stopwords.words("dutch")

    # pre processing data
    df = load_and_clean_data(filepath=args.filepath)
    df = sample_and_create_label(df=df, frac=args.frac)
    df = filter_labels(df=df)
    train_texts, train_labels, test_texts, test_labels = (
        generate_training_testing_data(df=df,
                                       split=args.split,
                                       columns=args.columns)
    )

    # After generating train_texts and train_labels
    train_texts, train_labels = balance_dataset(train_texts, train_labels, method='undersample')

    # Then proceed with model training
    pipeline = get_pipeline()
    parameters = get_parameters(optimized=True)
    grid_search = perform_grid_search_cv(train_texts=train_texts,
                                        train_labels=train_labels,
                                        pipeline=pipeline,
                                        parameters=parameters)

    # Concat column names used to create the filepath when saving files
    column_names = "_".join(args.columns).lower()

    # Export model
    model_filepath = f"{args.output_dir}/{column_names}_model.pkl"
    joblib.dump(grid_search, filename=model_filepath)
    logging.info(f"Exported model to: {model_filepath}")

    # Generate and export category paths
    category_paths = generate_category_paths(model=grid_search,
                                             columns=args.columns)
    category_paths_filepath = f"{args.output_dir}/{column_names}_slugs.pkl"
    joblib.dump(category_paths,
                filename=f"{args.output_dir}/"
                         f"{column_names}_slugs.pkl")
    logging.info(f"Exported category paths to: {category_paths_filepath}")

    # Validation
    test_predict = grid_search.predict(test_texts)
    precision, recall, accuracy, f1, balanced_acc = calculate_metrics(test_labels=test_labels, test_predict=test_predict)
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info(f"F1-score: {f1:.2f}")
    logging.info(f"Balanced Accuracy: {balanced_acc:.2f}")

    # Generate confusion matrix and save
    pdf_filepath = f"{args.output_dir}/{column_names}-matrix.pdf"
    csv_filepath = f"{args.output_dir}/{column_names}-matrix.csv"
    generate_and_save_confusion_matrix(model=grid_search,
                                       test_texts=test_texts,
                                       test_labels=test_labels,
                                       pdf_filepath=pdf_filepath,
                                       csv_filepath=csv_filepath)
    logging.info(f"Saved configuration matrix as PDF to: {pdf_filepath}")
    logging.info(f"Saved configuration matrix as CSV to: {csv_filepath}")

    # Save detailed validation
    if args.save_detailed_validation:
        csv_filepath = (f"{args.output_dir}/"
                        f"{column_names}_validation.csv")
        save_detailed_validation_results(test_texts=test_texts,
                                         test_predict=test_predict,
                                         test_labels=test_labels,
                                         csv_filepath=csv_filepath)
        logging.info("Saved detailed validation results as CSV to: "
                     f"{csv_filepath}")

    # Done
    logging.info('-' * 80)
    logging.info("Done!")
