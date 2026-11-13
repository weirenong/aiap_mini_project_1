# Standard library imports
import logging
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess news article share data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        df_cleaned = df.copy()

        # Drop ID and URL, which are unique identifiers for each article and carry no predictive value
        df_cleaned = df_cleaned.drop(columns=["ID", "URL"])
        logging.info(f"Shape after dropping ID and URL: {df_cleaned.shape}")

        # Replace nulls in content count features with 0, as a null means the article has none of that content
        replace_null_feature_list = [
            "num_hrefs",
            "num_self_hrefs",
            "num_imgs",
            "num_videos",
            "self_reference_min_shares",
            "self_reference_max_shares",
            "self_reference_avg_shares",
        ]
        for feature in replace_null_feature_list:
            df_cleaned[feature] = df_cleaned[feature].fillna(0)
        logging.info("Filled null content counts with 0.")

        # Replace null data_channel values with none, as the null channel has a distinct share distribution
        df_cleaned["data_channel"] = df_cleaned["data_channel"].fillna("none")
        logging.info(f"data_channel unique values: {df_cleaned['data_channel'].unique()}")

        # Replace negative values in kw features with 0, as share counts cannot be negative
        kw_feature_list = ["kw_min_min", "kw_avg_min", "kw_min_avg"]
        for feature in kw_feature_list:
            df_cleaned[feature] = df_cleaned[feature].clip(lower=0)
        logging.info("Clipped negative keyword share values at 0.")

        # Drop articles without content, as all text derived ratios are degenerate for them
        empty_articles = (df_cleaned["n_tokens_content"] == 0).sum()
        df_cleaned = df_cleaned[df_cleaned["n_tokens_content"] != 0]
        logging.info(
            f"Dropped {empty_articles} articles without content. Shape: {df_cleaned.shape}"
        )

        # Drop near constant columns that add no information
        drop_feature_list = ["kw_min_min", "kw_max_max", "n_non_stop_words"]
        df_cleaned = df_cleaned.drop(columns=drop_feature_list)
        logging.info(f"Dropped near constant columns: {drop_feature_list}")

        # Ensure the cleaned DataFrame only contains the columns specified in the config for numerical, nominal, passthrough features, and the target column
        df_cleaned = df_cleaned[
            [
                col
                for col in df_cleaned.columns
                if col
                in self.config["numerical_features"]
                + self.config["nominal_features"]
                + self.config["passthrough_features"]
                + [self.config["target_column"]]
            ]
        ]
        logging.info(f"Shape after selecting configured columns: {df_cleaned.shape}")

        logging.info("Data cleaning completed.")
        return df_cleaned

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical and nominal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]
        )

        transformers = [
            ("num", numerical_transformer, self.config["numerical_features"]),
            ("nom", nominal_transformer, self.config["nominal_features"]),
        ]

        # Only add the passthrough step when there are passthrough features configured
        if self.config["passthrough_features"]:
            transformers.append(
                ("pass", "passthrough", self.config["passthrough_features"])
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",  # Drop any column not declared in the config
            n_jobs=-1,
        )
        return preprocessor
