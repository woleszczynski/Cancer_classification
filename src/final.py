import os
import pandas as pd
import numpy as np
import warnings
from collections import Counter

# Podstawowe narzędzia ML / preprocessing z scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif  # używane do szybkiej selekcji cech

# Modele
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier

# Do nierównowagi klas
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # pipeline uwzględniający oversampling

warnings.filterwarnings("ignore")  # Aby log był czytelniejszy

# === Modele z ustalonymi hyperparametrami ===

fixed_models = {
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42, max_depth=10, criterion="gini"),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=100, max_depth=None, criterion="entropy"),
    # Liniowy SVM
    "SVM": LinearSVC(class_weight="balanced", random_state=42, C=10, max_iter=5000, tol=1e-3)
}


def need_smote(y, threshold=0.5):
    # Wykrywanie nierównowagi klas i czy stosować SMOTE
    counts = np.array(list(Counter(y).values()))
    if len(counts) <= 1:
        return False
    smallest = counts.min()
    largest = counts.max()
    return (smallest / largest) < threshold

def build_preprocessor(num_features, cat_features):
    # Spójny preprocessor danych wejściowych dla wszystkich modeli

    # Liczbowe uzupełnienie średnią + skalowanie
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    # Kategoryczne: uzupełnienie najczęstszą wartością i one-hot encoding
    try:
        onehot = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # kompatybilność ze starszymi wersjami sklearn
        onehot = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot)
    ])

    transformers = []
    if num_features:
        transformers.append(("num", numeric_transformer, num_features))
    if cat_features:
        transformers.append(("cat", categorical_transformer, cat_features))

    # ColumnTransformer scala przetworzone bloki, a resztę odrzuca
    return ColumnTransformer(transformers=transformers, remainder="drop")


# === Loadery dla zbiorów danych ===
# Każdy loader zwraca: X, y, listę cech numerycznych i kategorycznych

def load_seer(sample_frac=0.3, random_state=42):
    # SEER
    # Filtrowanie interesujących typów nowotworów, mapowanie dane tekstowych na liczby, usuwanie brakujących wartości
    # Opcjonalnie próbkowanie subset (sample_frac) dla szybkości testów

    df = pd.read_csv("datasets/seer_data.txt", sep="\t", low_memory=False)
    # Kody ICD dla interesujących typów nowotworów
    kody = ['C50', 'C34', 'C18', 'C19', 'C20', 'C16']

    def czy_interesujacy(kod):
        return any(str(kod).startswith(k) for k in kody)

    # Filtrowanie tylko rekordów z interesującymi lokalizacjami
    df = df[df['Primary Site - labeled'].apply(lambda x: czy_interesujacy(str(x).split('-')[0].strip()))].copy()

    # Zaklasyfikowanie kodów do nazw typów
    def klasyfikuj(kod):
        kod = str(kod).split('-')[0].strip()
        if kod.startswith('C50'):
            return 'Breast'
        elif kod.startswith('C34'):
            return 'Lung'
        elif kod.startswith('C16'):
            return 'Stomach'
        elif kod.startswith(('C18', 'C19', 'C20')):
            return 'Colorectal'
        else:
            return 'Other'

    df['CancerType'] = df['Primary Site - labeled'].apply(klasyfikuj)
    df = df[df['CancerType'] != 'Other'].copy()  # odrzucanie nieinteresujących

    # Zamiana przedziałów wiekowych na liczby
    wiek_map = {
        '0-1 years': 0.5, '1-4 years': 2.5, '5-9 years': 7, '10-14 years': 12, '15-19 years': 17,
        '20-24 years': 22, '25-29 years': 27, '30-34 years': 32, '35-39 years': 37, '40-44 years': 42,
        '45-49 years': 47, '50-54 years': 52, '55-59 years': 57, '60-64 years': 62, '65-69 years': 67,
        '70-74 years': 72, '75-79 years': 77, '80-84 years': 82, '85+ years': 87
    }
    df['Age recode with <1 year olds'] = df['Age recode with <1 year olds'].map(wiek_map)

    # Kodowanie zachowania nowotworu na liczby
    df['Behavior code ICD-O-3'] = df['Behavior code ICD-O-3'].replace(
        {'Benign': 0, 'Borderline malignancy': 1, 'In situ': 2, 'Malignant': 3}
    )

    # "Blank(s)" traktowane jako brak danych
    for col in ['CS tumor size (2004-2015)', 'CS mets at dx (2004-2015)', 'Laterality']:
        if col in df.columns:
            df[col] = df[col].replace('Blank(s)', np.nan)

    # Cechy wejściowe
    features = [
        'Age recode with <1 year olds',
        'CS tumor size (2004-2015)',
        'Sex',
        'Race recode (White, Black, Other)',
        'Laterality',
        'Behavior code ICD-O-3',
        'Histology ICD-O-2',
        'CS mets at dx (2004-2015)',
        'Year of diagnosis'
    ]
    # Usuwanie wierszy z brakami w kluczowych cechach
    df = df.dropna(subset=features + ['CancerType'])

    # Konwertowanie niektórych kolumn na numeryczne (żeby nie było stringów)
    for col in ['Age recode with <1 year olds', 'CS tumor size (2004-2015)', 'Behavior code ICD-O-3', 'Year of diagnosis']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[features]
    y = df['CancerType']

    # Próbkowanie jeśli duży zbiór (dla szybkiego debugowania/iteracji)
    if sample_frac < 1.0:
        X, _, y, _ = train_test_split(X, y, train_size=sample_frac, stratify=y, random_state=random_state)

    # Rozdzielanie cech numerycznych i kategorycznych
    numeric = ['Age recode with <1 year olds', 'CS tumor size (2004-2015)', 'Behavior code ICD-O-3', 'Year of diagnosis']
    categorical = list(set(features) - set(numeric))
    return X, y, numeric, categorical


def load_breast():
    # Breast Cancer

    df = pd.read_csv("datasets/breast_cancer.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")  # usuwa niepotrzebne kolumny
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # koduje etykiety
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = []
    return X, y, numeric, categorical


def load_tcga():
    # TCGA
    # Używany later gradient boosting przy searchu, żeby przyspieszyć obliczenia

    df = pd.read_csv("datasets/TCGA_combined_data.csv")
    X = df.drop(columns=['label'])
    y = df['label']
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = []
    return X, y, numeric, categorical


def load_lung():
    # Lung cancer
    # Kodowanie cech liczbowych

    df = pd.read_csv("datasets/lung_cancer_big.csv")
    le = LabelEncoder()
    if 'GENDER' in df.columns:
        df['GENDER'] = le.fit_transform(df['GENDER'])
    if 'LUNG_CANCER' in df.columns:
        df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
    X = df.drop(columns=['LUNG_CANCER'])
    y = df['LUNG_CANCER']
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = []
    return X, y, numeric, categorical


# === Główna pętla ===
def run_fixed():
    # Dla każdego zbioru danych ładuje dane, dzieli na trening/test, decyduje czy dać SMOTE
    # zapisuje wszystkie metryki

    os.makedirs("results/select", exist_ok=True)
    loaders = {
        "SEER": lambda: load_seer(sample_frac=0.3),
        "Breast": load_breast,
        "TCGA": load_tcga,
        "Lung": load_lung
    }

    for name, loader in loaders.items():
        print(f"\n\n=== ZBIÓR DANYCH {name} ===")
        X, y, num_features, cat_features = loader()

        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Sprawdza, czy warto zastosować SMOTE przy nierównowadze klas
        use_smote = need_smote(y_train)
        print(f"Rozkład klas: {Counter(y_train)} — {'Użyto SMOTE' if use_smote else 'Bez SMOTE'}\n")

        # Preprocessing
        preprocessor = build_preprocessor(num_features, cat_features)

        results = {}

        for model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "SVM"]:
            print(f"--- Trening {model_name} na {name} ---")

            if model_name == "Gradient Boosting":
                total_feat = len(num_features) + len(cat_features)
                k = min(200, total_feat) if total_feat > 0 else 0
                selector = SelectKBest(score_func=f_classif, k=k) if k > 0 else None

                gb = HistGradientBoostingClassifier(
                    random_state=42,
                    learning_rate=0.1,
                    max_depth=5,
                    max_iter=100,
                    early_stopping=True
                )

                steps = [("preprocessor", preprocessor)]
                if use_smote:
                    steps.append(("smote", SMOTE(random_state=42)))
                if selector is not None:
                    steps.append(("feature_select", selector))
                steps.append(("to_dense", FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X)))
                steps.append(("classifier", gb))

                pipe = ImbPipeline(steps) if use_smote else Pipeline(steps)

            else:
                model = fixed_models[model_name]
                if use_smote:
                    pipe = ImbPipeline([
                        ("preprocessor", preprocessor),
                        ("smote", SMOTE(random_state=42)),
                        ("classifier", model)
                    ])
                else:
                    pipe = Pipeline([
                        ("preprocessor", preprocessor),
                        ("classifier", model)
                    ])

            # Trening
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            # Ewaluacja
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            results[model_name] = {
                "report_df": pd.DataFrame(report).T,
                "confusion_df": pd.DataFrame(cm)
            }

        # Zapis wyników do Excela
        out_path = f"results/select/r_{name}_final.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for model_name, content in results.items():
                content["report_df"].to_excel(writer, sheet_name=f"{model_name}_metrics")
                content["confusion_df"].to_excel(writer, sheet_name=f"{model_name}_confusion")
        print(f"Ścieżka zapisu raportu: {out_path}")

if __name__ == "__main__":
    run_fixed()