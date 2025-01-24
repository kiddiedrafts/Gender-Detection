# Gender Detection from Social Media Profiles

This project uses machine learning to predict the gender of users based on their social media profiles (such as Twitter and Instagram). The model utilizes both structured features (e.g., age, follower count) and unstructured text data (e.g., biography, username) to make predictions. It employs XGBoost as the primary classification algorithm, with preprocessed textual data and feature engineering techniques.

## Dataset

The dataset consists of 8,000 rows and 10 columns, with a mix of structured and unstructured data. The target variable is the `gender` of the users (either male or female). Other columns provide information about the user's age, profile name, biography, follower count, and more.

**Note:** The dataset is proprietary, and thus, its publication is not permitted.

### Dataset Description
The dataset consists of social media records and includes the following columns:

| Column             | Description                             |
|--------------------|-----------------------------------------|
| gender             | The gender of the user (target variable) |
| age                | Age category (less than 18, 19-29, 30-40, 40+) |
| fullname           | Full name of the user on the social media profile |
| username           | Username of the user                    |
| biography          | User’s biography text                   |
| follower_count     | Number of followers                     |
| following_count    | Number of accounts the user follows     |
| is_business        | Whether the account is a business profile |
| is_verified        | Whether the account is verified         |
| is_private         | Whether the account is private          |

## Feature Engineering and Data Preprocessing

- **Text Preprocessing**: The `fullname`, `username`, and `biography` columns are cleaned by removing unwanted characters, digits, and extra spaces. All text is then converted to lowercase for consistency.
- **Emoji Count**: A new feature, `emoji_count`, is created by counting the number of emojis in the `biography` column.
- **Text Vectorization**: TF-IDF vectorization is applied to the combined text fields (`fullname`, `username`, `biography`) to convert the text into numerical features that can be used by the machine learning model.
- **Missing Data**: Missing values in the `is_business` column are filled with the most frequent value (mode).

## Model Training

- **XGBoost**: Two XGBoost classifier models are trained:
  1. A model using only the text features (via TF-IDF).
  2. A model incorporating both text and structured features (such as `age`, `follower_count`, `is_business`, etc.).

## Model Performance

The model achieved the following F1 scores:
- **Training Set:** 0.9068
- **Test Set:** 0.7923

The model demonstrates strong performance on the training data and reasonable performance on the test data, suggesting it generalizes well to unseen data.

## Requirements
To run this project, you'll need the following Python libraries:
- `pandas`
- `scikit-learn`
- `xgboost`
- `re`
- `emoji`
