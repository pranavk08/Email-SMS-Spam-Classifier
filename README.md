# SMS Spam Classifier

A machine learning-based web application that classifies SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and Naive Bayes classification.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)

## 🌟 Features

- **Real-time SMS Classification**: Instantly classify messages as spam or legitimate
- **Machine Learning Powered**: Uses Multinomial Naive Bayes algorithm
- **NLP Text Processing**: Advanced text preprocessing with NLTK
- **Beautiful UI**: Custom gradient background with Streamlit
- **High Accuracy**: Achieves ~97% accuracy on test data

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Training](#model-training)
- [Customization](#customization)
- [Technologies Used](#technologies-used)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or download this repository**

2. **Navigate to the project directory**
   ```bash
   cd sms-spam-project
   ```

3. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   ```

4. **Activate the virtual environment**
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

5. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

6. **Download NLTK data** (if not already downloaded)
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset** (`spam.csv`), which contains:
- **5,572 SMS messages**
- Labels: `ham` (legitimate) and `spam`
- Format: CSV with columns `v1` (label) and `v2` (message)

Place your `spam.csv` file in the project root directory.

## 📁 Project Structure

```
sms-spam-project/
│
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script
├── spam.csv              # Dataset (not included in repo)
├── vectorizer.pkl        # Trained TF-IDF vectorizer (generated)
├── model.pkl             # Trained Naive Bayes model (generated)
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🎯 Usage

### 1. Train the Model (First Time Setup)

Before running the app, train the model:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the Multinomial Naive Bayes classifier
- Save `vectorizer.pkl` and `model.pkl`
- Display test accuracy

**Expected Output:**
```
Test accuracy: 0.9740
Saved vectorizer.pkl and model.pkl
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Classify Messages

1. Enter any SMS message in the text input
2. Click the **Predict** button
3. See the classification result (Spam or Not Spam)

**Example Messages to Try:**
- **Spam**: "Congratulations! You've won a $1000 gift card. Click here to claim now!"
- **Not Spam**: "Hey, are we still meeting for lunch tomorrow?"

## 🔧 How It Works

### Text Preprocessing Pipeline

1. Lowercasing: Convert all text to lowercase
2. Tokenization: Split text into individual words using NLTK
3. Alphanumeric Filtering: Keep only alphanumeric tokens
4. Stopword Removal: Remove common English stopwords
5. Stemming: Reduce words to their root form using Porter Stemmer

### Machine Learning Pipeline

1. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
2. **Classification**: Multinomial Naive Bayes algorithm
3. **Prediction**: Classify new messages based on learned patterns

```
Input Message → Preprocessing → TF-IDF Vectorization → Model Prediction → Spam/Not Spam
```

## 🎓 Model Training

The `train_model.py` script performs the following steps:

1. **Load Dataset**: Read `spam.csv` with proper encoding (`latin-1`)
2. **Data Cleaning**: 
   - Map labels: `spam → 1`, `ham → 0`
   - Remove null values
3. **Text Transformation**: Apply NLP preprocessing
4. **Train-Test Split**: 80% training, 20% testing
5. **TF-IDF Vectorization**: Extract top 3000 features
6. **Model Training**: Fit Multinomial Naive Bayes classifier
7. **Evaluation**: Calculate accuracy on test set
8. **Serialization**: Save model and vectorizer as pickle files

### Model Performance

- **Algorithm**: Multinomial Naive Bayes
- **Accuracy**: ~97.4%
- **Features**: 3000 TF-IDF features
- **Training Time**: < 1 second

## 🎨 Customization

### Change Background Color

Edit `app.py` and modify the CSS section:

```python
# Solid color
st.markdown("""
    <style>
    .stApp {
        background-color: #3498db;  # Change this hex color
    }
    </style>
    """, unsafe_allow_html=True)
```

### Popular Color Options

- **Teal**: `#1abc9c`
- **Green**: `#2ecc71`
- **Blue**: `#3498db`
- **Purple**: `#9b59b6`
- **Orange**: `#e67e22`

### Change Model Parameters

Edit `train_model.py`:

```python
# Adjust TF-IDF features
tfidf = TfidfVectorizer(max_features=5000)  # Increase/decrease features

# Try different Naive Bayes parameters
model = MultinomialNB(alpha=0.5)  # Adjust smoothing parameter
```

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming language |
| **Streamlit** | Web application framework |
| **scikit-learn** | Machine learning library |
| **NLTK** | Natural language processing |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Pickle** | Model serialization |

## 📝 Code Explanation

### `app.py` - Main Application

- Loads pre-trained model and vectorizer
- Creates Streamlit UI with custom styling
- Processes user input through the same preprocessing pipeline
- Displays prediction results

### `train_model.py` - Model Training

- Handles dataset loading with proper encoding
- Implements text preprocessing function
- Trains and evaluates the classification model
- Saves trained artifacts for deployment

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Activate virtual environment and install dependencies
```bash
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'spam.csv'"
**Solution**: Ensure `spam.csv` is in the project root directory

### Issue: "NotFittedError: This MultinomialNB instance is not fitted"
**Solution**: Train the model first
```bash
python train_model.py
```

### Issue: "UnicodeDecodeError when loading CSV"
**Solution**: Already fixed in code using `encoding='latin-1'`

## 📈 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add confidence score display
- [ ] Create REST API endpoint
- [ ] Add batch prediction capability
- [ ] Deploy to cloud (Streamlit Cloud, Heroku, AWS)



## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

