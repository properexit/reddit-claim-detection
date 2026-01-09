import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from src.baselines_utils import load_fasttext_subset, get_mean_vector

# 1. Prepare Data
df = pd.read_csv("data/raw/labels_v6.csv")
# For Multiclass: 0=No Claim, 1=Explicit, 2=Implicit
df['multiclass_label'] = 0
df.loc[df['explicit'] == 1, 'multiclass_label'] = 1
df.loc[(df['claim'] == 1) & (df['explicit'] == 0), 'multiclass_label'] = 2

# 2. Build Vocab & Load FastText
all_words = set(" ".join(df['text'].str.lower()).split())
word_to_vec = load_fasttext_subset("embeddings/crawl-300d-2M.vec", all_words)

# 3. Vectorize
X = np.array([get_mean_vector(t, word_to_vec) for t in df['text']])
y_bin = df['claim'].values
y_multi = df['multiclass_label'].values

# 4. Train/Test Split & Evaluate
# (Logic for LR and GaussianNB goes here using sklearn.model_selection)