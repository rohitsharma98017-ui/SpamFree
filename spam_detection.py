import sys
import os
import re
import urllib.request
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
DATA_FILE_PATH = "sms_spam_data.csv"
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
BACKUP_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/sms_spam.csv"


def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATA_FILE_PATH):
        print(f"[+] Found cached dataset locally at '{DATA_FILE_PATH}'.")
        df = pd.read_csv(DATA_FILE_PATH)
        return df[['text', 'label']]

    print("[*] Downloading SMS Spam dataset from primary URL...")
    for url, sep, header_spec, label_col, text_col in [
        (DATASET_URL, '\t', None, 0, 1),
        (BACKUP_URL, ',', 0, 'type', 'text')
    ]:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
                if header_spec is None:
                    raw_df = pd.read_csv(StringIO(content), sep=sep, header=None)
                    df = pd.DataFrame({
                        'label': raw_df[label_col].map(lambda x: 1 if str(x).lower().strip() == 'spam' else 0),
                        'text': raw_df[text_col].astype(str)
                    })
                else:
                    raw_df = pd.read_csv(StringIO(content), sep=sep)
                    df = pd.DataFrame({
                        'label': raw_df[label_col].map(lambda x: 1 if str(x).lower().strip() == 'spam' else 0),
                        'text': raw_df[text_col].astype(str)
                    })

                df.to_csv(DATA_FILE_PATH, index=False)
                print(f"[+] Successfully downloaded and cached {len(df)} messages to '{DATA_FILE_PATH}'.")
                return df[['text', 'label']]
        except Exception as e:
            print(f"[-] Download attempt from '{url}' failed: {e}")

    print("[!] Network unavailable. Initializing curated offline fallback dataset...")
    offline_samples = [
        ("Hey, are we still meeting for lunch at 12:30 today?", 0),
        ("Can you please send me the notes from yesterday's meeting?", 0),
        ("Your Amazon order #402-1928 has been dispatched and arrives tomorrow.", 0),
        ("Don't forget mom's birthday dinner on Friday night at 7pm.", 0),
        ("The quarterly project report has been uploaded to the shared team drive.", 0),
        ("Are you coming to the gym later or resting today?", 0),
        ("Let me know when you get home safely, drive carefully!", 0),
        ("Attached is the updated invoice for consulting services rendered.", 0),
        ("Doctor appointment confirmed for Monday 10:00 AM.", 0),
        ("Your GitHub verification one-time passcode is 482910.", 0),
        ("Can we reschedule our one-on-one call to 3 PM?", 0),
        ("Thanks for the quick response, really appreciate your help!", 0),
        ("Please review the pull request when you have 10 minutes.", 0),
        ("Flight check-in is now open for your upcoming trip to Chicago.", 0),
        ("Hey man, do you have the professor's email address?", 0),
        ("Dinner was amazing tonight, thank you so much for hosting!", 0),
        ("The grocery list: milk, eggs, sourdough bread, and apples.", 0),
        ("Your library book 'Designing Data-Intensive Applications' is due in 3 days.", 0),
        ("Good morning team, reminder that standup starts in 15 minutes.", 0),
        ("I will be working from home today due to a mild fever.", 0),
        ("CONGRATULATIONS! You have won a $1,000 Walmart Gift Card! Call 0800123456 now to claim your prize!", 1),
        ("URGENT! Your mobile number won 500,000 GBP in the UK National Lottery. Reply with your bank account details.", 1),
        ("FREE ENTRY into our weekly competition to win a brand new iPhone 15 Pro! Text WIN to 80085 immediately.", 1),
        ("Get rich quick with guaranteed 200% weekly crypto returns! Click http://bit.ly/fast-crypto-now to deposit.", 1),
        ("LOAN APPROVED: Instant cash up to $25,000 with NO credit check! Apply today at http://fast-cash-loans.net", 1),
        ("Hot singles in your local area want to chat tonight! Call 09061743812 now. Calls cost 1.50/min.", 1),
        ("URGENT SECURITY ALERT: Your PayPal account has been suspended! Verify your password at http://paypal-security-login.com", 1),
        ("Exclusive luxury replica Rolex watches at 90% discount! Limited clearance sale ends midnight.", 1),
        ("You have an unclaimed cash refund of $4,500 waiting from IRS. Click link to submit your Social Security Number.", 1),
        ("FINAL NOTICE: Your car warranty has expired. Press 1 to speak with an agent before your coverage is cancelled.", 1),
        ("WINNER! As a valued customer you have been selected for a free holiday cruise to Bahamas! Call 1-800-CRUISE.", 1),
        ("Work from home and earn $500 to $1,500 daily stuffing envelopes! No experience needed. Reply YES to start.", 1),
        ("Refill your prescription pills online with no doctor prescription needed! Lowest prices guaranteed.", 1),
        ("Congratulations! You qualify for debt relief consolidation. Cut credit card bills in half today! Call now.", 1),
        ("SPECIAL PROMO: Buy 1 Get 3 Free Swiss designer sunglasses! Today only at http://shady-glasses-deals.com", 1),
    ] * 20
    
    df = pd.DataFrame(offline_samples, columns=['text', 'label'])
    df.to_csv(DATA_FILE_PATH, index=False)
    return df


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_feature_pipeline(X_train: pd.Series, X_test: pd.Series, max_features: int = 1500):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english'
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return vectorizer, X_train_tfidf, X_test_tfidf


def train_linear_regression(X_train_tfidf, y_train, X_test_tfidf):
    model = LinearRegression()
    model.fit(X_train_tfidf, y_train)
    y_pred_continuous = model.predict(X_test_tfidf)
    y_pred_class = (y_pred_continuous >= 0.5).astype(int)
    return model, y_pred_continuous, y_pred_class


def demonstrate_linear_regression_flaws(y_pred_continuous, y_test):
    min_val = float(np.min(y_pred_continuous))
    max_val = float(np.max(y_pred_continuous))
    count_below_zero = int(np.sum(y_pred_continuous < 0.0))
    count_above_one = int(np.sum(y_pred_continuous > 1.0))
    total_test = len(y_pred_continuous)
    pct_invalid = ((count_below_zero + count_above_one) / total_test) * 100

    print("\n" + "=" * 78)
    print("EMPIRICAL PROOF: WHY LINEAR REGRESSION FAILS AS A CLASSIFIER")
    print("=" * 78)
    print(f"Total test samples evaluated: {total_test}")
    print(f"[-] Minimum continuous prediction: {min_val:+.4f}  (VIOLATION: Probability < 0)")
    print(f"[-] Maximum continuous prediction: {max_val:+.4f}  (VIOLATION: Probability > 1)")
    print(f"[-] Samples predicting P < 0:      {count_below_zero:4d} ({count_below_zero/total_test*100:5.2f}%)")
    print(f"[-] Samples predicting P > 1:      {count_above_one:4d} ({count_above_one/total_test*100:5.2f}%)")
    print(f"[!] Total probability violations:  {count_below_zero + count_above_one:4d} ({pct_invalid:5.2f}%)")
    print("-" * 78)
    print("Theoretical Analysis:")
    print("  'Linear Regression assumes continuous Gaussian targets. When applied to 0/1")
    print("   binary targets with MSE loss, it treats extreme features as outliers,")
    print("   violates probability axioms [0, 1], and severely degrades decision boundaries.'")
    print("=" * 78 + "\n")


def train_logistic_regression(X_train_tfidf, y_train, X_test_tfidf):
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    model.fit(X_train_tfidf, y_train)
    y_pred_class = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    return model, y_pred_proba, y_pred_class


def train_naive_bayes(X_train_tfidf, y_train, X_test_tfidf):
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    y_pred_class = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    return model, y_pred_proba, y_pred_class


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cm": cm
    }


def print_comparison_table(results: list):
    print("=" * 86)
    print("MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 86)
    header = f"{'Model Name':<32} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'FP (Costly)':<10}"
    print(header)
    print("-" * 86)
    for res in results:
        row = (
            f"{res['model_name']:<32} | "
            f"{res['accuracy']*100:6.2f}%   | "
            f"{res['precision']*100:6.2f}%   | "
            f"{res['recall']*100:6.2f}%   | "
            f"{res['f1']*100:6.2f}%   | "
            f"{res['fp']:<10d}"
        )
        print(row)
    print("=" * 86)
    
    print("\nDetailed Confusion Matrices Breakdown:")
    for res in results:
        print(f"\n[{res['model_name']}]")
        print(f"  True Negatives  (Ham correctly identified):  {res['tn']:4d}")
        print(f"  False Positives (Ham flagged as SPAM - FP):   {res['fp']:4d}  <-- CRITICAL ERROR")
        print(f"  False Negatives (Spam landed in INBOX - FN):  {res['fn']:4d}")
        print(f"  True Positives  (Spam correctly caught):      {res['tp']:4d}")


def explain_precision_importance():
    print("""
================================================================================
CRITICAL METRIC ANALYSIS: WHY PRECISION IS ESSENTIAL IN SPAM FILTERING
================================================================================
In classification problems, the cost of Type I vs Type II errors is asymmetric:

1. False Positive (Type I Error) - Predicted Spam, Actually Ham:
   - A legitimate email (e.g., job offer, medical test result, two-factor OTP code,
     university admission, or client contract) is silently rerouted to Spam/Trash.
   - Cost: CATASTROPHIC. The user misses vital life or business communications.

2. False Negative (Type II Error) - Predicted Ham, Actually Spam:
   - A spam message (e.g., discount pill flyer, fake lottery notification) slips
     through into the primary inbox.
   - Cost: MINOR INCONVENIENCE. The user takes 1 second to click 'Delete'.

Mathematical Definition of Precision:
   Precision = True Positives / (True Positives + False Positives)

Therefore:
   Maximizing Precision directly minimizes False Positives (FP -> 0).
   In production email filters (e.g., Gmail, Outlook), engineers tune the classification
   threshold to achieve Precision > 99.5%, willingly tolerating slightly lower Recall
   to guarantee that legitimate emails never get lost in the spam folder!
================================================================================
""")


def predict_message(text: str, vectorizer: TfidfVectorizer, model, model_name: str = "Model", threshold: float = 0.5) -> dict:
    cleaned = clean_text(text)
    vec_features = vectorizer.transform([cleaned])
    if hasattr(model, "predict_proba"):
        proba_spam = float(model.predict_proba(vec_features)[0, 1])
        predicted_label = 1 if proba_spam >= threshold else 0
    else:
        raw_score = float(model.predict(vec_features)[0])
        proba_spam = raw_score
        predicted_label = 1 if raw_score >= threshold else 0

    class_name = "SPAM" if predicted_label == 1 else "HAM (Legitimate)"
    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "model": model_name,
        "prediction": class_name,
        "is_spam": bool(predicted_label),
        "score": proba_spam
    }


def run_sample_predictions(vectorizer, lin_model, log_model, nb_model):
    test_samples = [
        "Hey! Are we still grabbing dinner together tonight around 7?",
        "URGENT: Your account has been compromised. Verify your bank password now at http://secure-banking-alert.com",
        "Your Google verification code is 849201. Never share this code with anyone.",
        "CONGRATULATIONS!! You have won a $1,000 Amazon Gift Card! Text CLAIM to 88202 to collect immediately!",
        "Can you send over the updated slides for the board meeting tomorrow morning?",
        "Double your money in 48 hours with our automated Bitcoin trading bot! 100% risk free guaranteed returns!"
    ]

    print("\n" + "=" * 86)
    print("LIVE INFERENCE ON SAMPLE MESSAGES")
    print("=" * 86)
    for idx, sample in enumerate(test_samples, 1):
        print(f"\nSample #{idx}: \"{sample}\"")
        log_res = predict_message(sample, vectorizer, log_model, model_name="Logistic Regression")
        nb_res = predict_message(sample, vectorizer, nb_model, model_name="Multinomial Naive Bayes")
        lin_res = predict_message(sample, vectorizer, lin_model, model_name="Linear Regression")
        print(f"  -> Logistic Reg:     {log_res['prediction']:<16} [P(Spam) = {log_res['score']*100:5.1f}%]")
        print(f"  -> Naive Bayes:      {nb_res['prediction']:<16} [P(Spam) = {nb_res['score']*100:5.1f}%]")
        print(f"  -> Linear Reg (OLS): {lin_res['prediction']:<16} [Score   = {lin_res['score']:+6.3f} (Continuous)]")
    print("=" * 86)


def main():
    print("""
================================================================================
   SPAM EMAIL DETECTION: LINEAR REGRESSION VS LOGISTIC REGRESSION & NAIVE BAYES
================================================================================
""")
    df = load_dataset()
    print(f"[i] Dataset Summary:")
    print(f"    - Total records: {len(df)}")
    print(f"    - Ham count:     {(df['label'] == 0).sum()} ({(df['label'] == 0).mean()*100:.1f}%)")
    print(f"    - Spam count:    {(df['label'] == 1).sum()} ({(df['label'] == 1).mean()*100:.1f}%)")

    print("\n[*] Preprocessing message text (lowercase, regex non-alphabet cleaning)...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    print(f"[*] Splitting dataset: 80% train / 20% test (Stratified, random_state={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['label'],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )
    print(f"    - Training samples: {len(X_train)} ({y_train.sum()} Spam)")
    print(f"    - Testing samples:  {len(X_test)} ({y_test.sum()} Spam)")

    print("\n[*] Fitting TF-IDF Vectorizer on training set (max_features=1500, stop_words='english')...")
    vectorizer, X_train_tfidf, X_test_tfidf = build_feature_pipeline(X_train, X_test, max_features=1500)
    print(f"[+] Vocabulary size: {len(vectorizer.get_feature_names_out())} features extracted.")

    print("\n[*] Training Model 1: Linear Regression (OLS on binary 0/1 labels)...")
    lin_model, lin_pred_cont, lin_pred_class = train_linear_regression(X_train_tfidf, y_train, X_test_tfidf)
    demonstrate_linear_regression_flaws(lin_pred_cont, y_test)

    print("[*] Training Model 2: Logistic Regression (Sigmoid + Log-Loss)...")
    log_model, log_pred_proba, log_pred_class = train_logistic_regression(X_train_tfidf, y_train, X_test_tfidf)

    print("[*] Training Model 3: Multinomial Naive Bayes (Laplace alpha=1.0)...")
    nb_model, nb_pred_proba, nb_pred_class = train_naive_bayes(X_train_tfidf, y_train, X_test_tfidf)

    results = [
        evaluate_model(y_test, lin_pred_class, "1. Linear Regression (Threshold 0.5)"),
        evaluate_model(y_test, log_pred_class, "2. Logistic Regression (Sigmoid)"),
        evaluate_model(y_test, nb_pred_class, "3. Multinomial Naive Bayes (Baseline)")
    ]
    print_comparison_table(results)

    explain_precision_importance()

    run_sample_predictions(vectorizer, lin_model, log_model, nb_model)

    if "--interactive" in sys.argv:
        print("\n" + "=" * 70)
        print("INTERACTIVE PREDICTION MODE (Type 'exit' or 'quit' to stop)")
        print("=" * 70)
        while True:
            try:
                user_msg = input("\nEnter custom email/SMS text: ").strip()
                if user_msg.lower() in ['exit', 'quit', '']:
                    break
                pred = predict_message(user_msg, vectorizer, log_model, model_name="Logistic Regression")
                print(f"Result: {pred['prediction']} (Confidence: {pred['score']*100:.2f}%)")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting interactive mode. Goodbye!")


if __name__ == "__main__":
    main()
