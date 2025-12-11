# -*- coding: utf-8 -*-

!pip install scikit-learn
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

DATA_DIR = "/content/drive/MyDrive/mtllm/data"  # adjust path
CODE_DIR = "/content/drive/MyDrive/mtllm/code"  # where validation.py lives

import sys
sys.path.append(CODE_DIR)

import os, glob, random
import validation  # this is validation.py
from validation import Article, validate_articles, check_citations

from sklearn.feature_extraction.text import TfidfVectorizer

# 2.1 Load a subset of raw texts to fit the vectorizer
def collect_corpus_texts(data_dir, limit=5000):
    texts = []
    for i, path in enumerate(glob.glob(os.path.join(data_dir, "*.txt"))):
        if i >= limit:
            break
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return texts

corpus_texts = collect_corpus_texts(DATA_DIR, limit=5000)

vectorizer = TfidfVectorizer(max_features=2000)
vectorizer.fit(corpus_texts)

# 2.2 Monkey-patch embed_text inside the validation module
import numpy as np

def embed_text(text: str) -> np.ndarray:
    return vectorizer.transform([text]).toarray()[0]

validation.embed_text = embed_text

def load_articles_from_dir(data_dir, limit=5000):
    articles = []
    paths = glob.glob(os.path.join(data_dir, "*.txt"))
    random.shuffle(paths)  # random subset

    for i, path in enumerate(paths):
        if i >= limit:
            break
        with open(path, encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        lines = txt.strip().splitlines()
        title = lines[0].strip() if lines else os.path.basename(path)
        abstract = " ".join(lines[1:10])  # first few lines as pseudo-abstract

        pmcid = os.path.splitext(os.path.basename(path))[0]

        art = Article(
            pmcid=pmcid,
            doi=None,
            title=title[:200],
            abstract=abstract[:3000],
            journal="SampleCorpus",
            year=2020,
            citation_count=0,
            is_peer_reviewed=False,
            is_retracted=False,
        )
        articles.append(art)

    return articles

articles = load_articles_from_dir(DATA_DIR, limit=3000)
len(articles)

query = "antibiotic resistance in gram-negative bacteria"

validated = validate_articles(query, articles)
len(validated), len(articles)

import matplotlib.pyplot as plt

scores = [a.validation_score for a in validated]

plt.figure()
plt.hist(scores, bins=20)
plt.xlabel("Validation Score")
plt.ylabel("Frequency")
plt.title("Distribution of Article Validation Scores (Real Data)")
plt.show()

top5 = sorted(validated, key=lambda a: a.validation_score, reverse=True)[:5]
for a in top5:
    print(a.pmcid, a.validation_score, a.title)
    print()

# Use top 3 validated articles as "cited" papers
top3 = sorted(validated, key=lambda a: a.validation_score, reverse=True)[:3]

answer_body = f"""
Several studies have investigated this problem [1, 2]. Recent work emphasizes the
role of mobile genetic elements [3].
"""

# We'll pretend these are the reference list entries:
ref_lines = []
for i, art in enumerate(top3, start=1):
    ref_lines.append(f"[{i}] {art.title}. SampleCorpus, 2020. PMCID {art.pmcid}.")
ref_text = "\n".join(ref_lines)

print(ref_text)

# Sort scores and take the top 50
sorted_scores = np.sort(scores)
top_k = sorted_scores[-50:]

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(top_k) + 1), top_k, marker="o")
plt.xlabel("Rank (1 = highest)")
plt.ylabel("Validation score")
plt.title("Top 50 Article Validation Scores")
plt.tight_layout()

plt.savefig("/content/drive/MyDrive/mtllm/validation_top50.png", dpi=300, bbox_inches="tight")
plt.show()

# Define bands
import numpy as np

scores = np.array(scores)
low_mask = scores < 0.20
mid_mask = (scores >= 0.20) & (scores < 0.28)
high_mask = scores >= 0.28

counts = [low_mask.sum(), mid_mask.sum(), high_mask.sum()]
labels = ["Low", "Medium", "High"]

plt.figure(figsize=(6, 4))
plt.bar(labels, counts)
plt.xlabel("Confidence band")
plt.ylabel("Number of articles")
plt.title("Articles per Validation Band")
plt.tight_layout()

plt.savefig("/content/drive/MyDrive/mtllm/validation_bands.png", dpi=300, bbox_inches="tight")
plt.show()

counts

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# 1.1 Recompute document vectors from title + abstract
texts = [a.title + " " + a.abstract for a in articles]
doc_mat = vectorizer.transform(texts)        # sparse (n_docs, n_features)
doc_mat_dense = doc_mat.toarray()            # dense, needed for Pearson

# 1.2 Query vector
query_vec_sparse = vectorizer.transform([query])
query_vec = query_vec_sparse.toarray()[0]

n_docs, dim = doc_mat_dense.shape
print("Doc matrix shape:", doc_mat_dense.shape)

# 2.1 Cosine similarity (sklearn)
cos_sims = cosine_similarity(
    query_vec_sparse,    # shape (1, dim)
    doc_mat              # shape (n_docs, dim)
)[0]                     # shape (n_docs,)

# 2.2 Dot product
dot_sims = doc_mat_dense @ query_vec    # shape (n_docs,)

# 2.3 Pearson correlation coefficient
pearson_sims = np.zeros(n_docs)

for i in range(n_docs):
    v = doc_mat_dense[i]
    # Handle all-zero vectors safely
    if np.all(v == 0) or np.all(query_vec == 0):
        pearson_sims[i] = 0.0
    else:
        r, _ = pearsonr(query_vec, v)
        # Replace NaN (which can happen if variance is 0) with 0
        if np.isnan(r):
            r = 0.0
        pearson_sims[i] = r

print("Cosine:   min=%.4f max=%.4f mean=%.4f" % (cos_sims.min(), cos_sims.max(), cos_sims.mean()))
print("Dot:      min=%.4f max=%.4f mean=%.4f" % (dot_sims.min(), dot_sims.max(), dot_sims.mean()))
print("Pearson:  min=%.4f max=%.4f mean=%.4f" % (pearson_sims.min(), pearson_sims.max(), pearson_sims.mean()))
print(np.std(cos_sims), np.std(dot_sims), np.std(pearson_sims))

import os

out_dir = "/content/drive/MyDrive/mtllm/mtllm_figs_metrics"
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(6,4))
plt.hist(cos_sims, bins=25)
plt.title("Cosine Similarity Distribution")
plt.xlabel("Cosine similarity")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "cosine_hist.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(dot_sims, bins=25)
plt.title("Dot Product Similarity Distribution")
plt.xlabel("Dot product")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "dot_hist.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(pearson_sims, bins=25)
plt.title("Pearson Correlation Distribution")
plt.xlabel("Pearson correlation coefficient")
plt.ylabel("Number of documents")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pearson_hist.png"), dpi=300, bbox_inches="tight")
plt.show()

def band_counts(values, name):
    values = np.array(values)

    low_thr = np.quantile(values, 0.33)
    high_thr = np.quantile(values, 0.66)

    bands = np.empty(len(values), dtype=object)
    bands[values < low_thr] = "Low"
    bands[(values >= low_thr) & (values < high_thr)] = "Medium"
    bands[values >= high_thr] = "High"

    unique, counts = np.unique(bands, return_counts=True)
    print(f"\n{name} bands (thresholds: low<{low_thr:.4f}, high>{high_thr:.4f}):")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} ({c/len(values)*100:.1f}%)")

    return bands, low_thr, high_thr

cos_bands, cos_low, cos_high = band_counts(cos_sims, "Cosine")
dot_bands, dot_low, dot_high = band_counts(dot_sims, "Dot product")
pearson_bands, pearson_low, pearson_high = band_counts(pearson_sims, "Pearson")

def plot_band_bar(bands, name, filename):
    labels, counts = np.unique(bands, return_counts=True)
    plt.figure(figsize=(4,3))
    plt.bar(labels, counts)
    plt.title(f"{name} – Band counts")
    plt.xlabel("Band")
    plt.ylabel("Number of documents")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight") # Save before showing
    plt.show()

plot_band_bar(cos_bands, "Cosine", "/content/drive/MyDrive/mtllm/mtllm_figs_metrics/cosine_bands.png")
plot_band_bar(dot_bands, "Dot product", "/content/drive/MyDrive/mtllm/mtllm_figs_metrics/dot_bands.png")
plot_band_bar(pearson_bands, "Pearson", "/content/drive/MyDrive/mtllm/mtllm_figs_metrics/pearson_bands.png")