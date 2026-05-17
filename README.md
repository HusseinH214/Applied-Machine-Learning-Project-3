**Project**: Clustering of News Articles
- **Description:**: Unsupervised clustering pipeline that combines five topical datasets (business, education, entertainment, sports, technology), preprocesses text, converts it with TF-IDF + Truncated SVD, and compares clustering algorithms (KMeans, DBSCAN, Agglomerative).

**Data**
- **Source files:**: [data/business_data.csv](data/business_data.csv), [data/education_data.csv](data/education_data.csv), [data/entertainment_data.csv](data/entertainment_data.csv), [data/sports_data.csv](data/sports_data.csv), [data/technology_data.csv](data/technology_data.csv)
- **Combined / cleaned output:**: [data/cleaned_data.csv](data/cleaned_data.csv)

**Notebook**
- **Main analysis:**: See [clustering.ipynb](clustering.ipynb) for the full preprocessing, modeling, evaluation and plots.

**Pipeline Overview**
- **Preprocessing:**: combine datasets, drop metadata columns (`url`, `category`), concatenate `headlines`, `description`, and `content` into a `Text` field; lowercase, remove stopwords/punctuation and lemmatize using spaCy.
- **Feature transformation:**: `TfidfVectorizer(max_df=0.95)` then dimensionality reduction with `TruncatedSVD` (experiment used `n_components=300`) followed by `Normalizer`.
- **Clustering models evaluated:**: `KMeans` (elbow method, k≈5 used), `DBSCAN` (eps tuning via nearest-neighbors plot, eps=0.9 used), `AgglomerativeClustering` (dendrogram used to choose clusters).
- **Evaluation metrics:**: silhouette score and Davies–Bouldin score are computed to compare cluster quality.

**Key parameters & decisions**
- **TF-IDF:**: `max_df=0.95`
- **SVD components:**: `300`
- **KMeans:**: `n_clusters=5`, `random_state=6`
- **DBSCAN:**: `eps=0.9`, `min_samples=5` 

**Requirements & Setup**
Install dependencies and the spaCy English model before running the notebook.

```bash
python -m spacy download en_core_web_sm
```

```bash
pip install pandas numpy scikit-learn spacy matplotlib seaborn ipywidgets
```

**Run / Usage**
- Open the notebook [clustering.ipynb](clustering.ipynb) in Jupyter or VS Code and run the cells sequentially. The notebook saves a cleaned dataset to [data/cleaned_data.csv](data/cleaned_data.csv) to avoid repeated preprocessing.


