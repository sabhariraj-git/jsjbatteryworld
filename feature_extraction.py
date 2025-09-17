import pandas as pd
import re

# Load dataset
df = pd.read_csv("urls.csv")

def extract_features(url: str) -> dict:
    url = str(url).strip()
    url_lower = url.lower().strip()
    features = {}
    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["has_ip"] = 1 if re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url) else 0
    suspicious_words = ["login","signin","sign-in","authenticate","account","secure","verification","verify","confirm","confirmation","update","reset","password","credential",
                        "bank","banking","paypal","appleid","mastercard","visa","stripe","payment",
                        "billing","invoice","transaction","urgent","alert","locked","suspended",
                        "otp","token","authorize","claim","refund","unclaimed","verify-email"]
    features["suspicious_keywords"] = sum(1 for kw in suspicious_words if kw in url_lower)
    return features

# Extract features from dataset
feature_rows = []
for row in df.itertuples(index=False):
    feats = extract_features(row.url)
    feats["url"] = row.url
    feats["label"] = row.label
    feature_rows.append(feats)

features_df = pd.DataFrame(feature_rows)

# Display full DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("Features extracted from dataset:")
print(features_df)

# Save to CSV
features_df.to_csv("features_output.csv", index=False)
print("\nSaved features to 'features_output.csv'.")
