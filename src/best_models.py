import os
import pandas as pd
import numpy as np
import warnings
from collections import Counter
import time

# Podstawowe narzędzia ML / preprocessing z scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif  # używane do szybkiej selekcji cech

# Modele
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Do nierównowagi klas
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # pipeline wspierający oversampling

warnings.filterwarnings("ignore")  # Aby log był czytelniejszy

# === Modele z ustalonymi hyperparametrami ===
# Porównywanie tych samych przestrzeni dla każdego modelu i datasetu (z wyjątkami później)
base_models_and_params = {
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {
            "classifier__max_depth": [3, 5, 10, None],  # kontrola złożoności drzewa
            "classifier__criterion": ["gini", "entropy"]  # różne kryteria podziału
        }
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {
            "classifier__n_estimators": [50, 100],  # liczba drzew
            "classifier__max_depth": [5, 10, None],  # głębokość drzew
            "classifier__criterion": ["gini", "entropy"]
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "classifier__n_estimators": [50, 100],  # liczba słabych estymatorów
            "classifier__learning_rate": [0.01, 0.1],  # krok gradiantu
            "classifier__max_depth": [3, 5]  # złożoność każdej sekwencji
        }
    ),
    "SVM": (
        SVC(class_weight="balanced", probability=True, random_state=42),
        {
            "classifier__C": [0.1, 1, 10],  # regularyzacja
            "classifier__kernel": ["linear", "rbf"],  # liniowy vs nieliniowy
            "classifier__gamma": ["scale", "auto"]  # parametr jądra dla rbf
        }
    )
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
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    transformers = []
    if num_features:
        transformers.append(("num", numeric_transformer, num_features))
    if cat_features:
        transformers.append(("cat", categorical_transformer, cat_features))
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
    df = df[df['CancerType'] != 'Other'].copy() # odrzucanie nieinteresujących

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
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore") # usuwa niepotrzebne kolumny
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) # koduje etykiety
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
def run_all():
    # Dla każdego zbioru danych ładuje dane, dzieli na trening/test, decyduje czy dać SMOTE
    # buduje pipeline, robi randomized search po hiperparametrach, zapisuje wszystkie metryki, najlepsze parametry i czasy

    os.makedirs("results", exist_ok=True)
    dataset_loaders = {
        "SEER": lambda: load_seer(sample_frac=0.3),  # skracamy SEER przez sample_frac do 30% dla czasu
        "Breast": load_breast,
        "TCGA": load_tcga,
        "Lung": load_lung
    }

    for ds_name, loader in dataset_loaders.items():
        print(f"\n\n=== ZBIÓR DANYCH {ds_name} ===")
        X, y, num_features, cat_features = loader()

        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Sprawdza, czy warto zastosować SMOTE przy nierównowadze klas
        use_smote = need_smote(y_train)
        print(f"Rozkład klas: {Counter(y_train)} — {'Użyto SMOTE' if use_smote else 'Bez SMOTE'}\n")

        results = {}

        # Pętla po modelach i odpowiadających przestrzeniach hiperparametrów
        for model_name, (base_model, base_param_dist) in base_models_and_params.items():
            # SEER - Używane jest liniowe SVM dla wydajności ze względu na dużą ilość próbek
            if ds_name == "SEER" and model_name == "SVM":
                model = LinearSVC(class_weight="balanced", random_state=42, max_iter=5000, tol=1e-3)
                param_dist = {"classifier__C": [0.1, 1, 10]}
            # TCGA - HistGradientBoosting (selekcja cech)
            elif ds_name == "TCGA" and model_name == "Gradient Boosting":
                model = HistGradientBoostingClassifier(random_state=42, early_stopping=True)
                param_dist = {
                    "classifier__max_iter": [100, 200],  # liczba iteracji
                    "classifier__learning_rate": [0.01, 0.1],   #uczenie
                    "classifier__max_depth": [3, 5] #głębokość
                }
            else:
                model = base_model
                param_dist = base_param_dist

            print(f"\n--- {model_name} na {ds_name} ---")
            preprocessor = build_preprocessor(num_features, cat_features)

            # Budowanie pipelineu z uwzględnieniem SMOTE i wyjątków
            if ds_name == "TCGA" and model_name == "Gradient Boosting":
                # TCGA selekcja cech przed HistGradientBoosting
                selector = SelectKBest(score_func=f_classif, k=200)
                if use_smote:
                    pipeline = ImbPipeline([
                        ("preprocessor", preprocessor),
                        ("smote", SMOTE(random_state=42)),
                        ("feature_select", selector),
                        ("classifier", model)
                    ])
                else:
                    pipeline = Pipeline([
                        ("preprocessor", preprocessor),
                        ("feature_select", selector),
                        ("classifier", model)
                    ])
            else:
                if use_smote:
                    pipeline = ImbPipeline([
                        ("preprocessor", preprocessor),
                        ("smote", SMOTE(random_state=42)),
                        ("classifier", model)
                    ])
                else:
                    pipeline = Pipeline([
                        ("preprocessor", preprocessor),
                        ("classifier", model)
                    ])

            # Losowe przeszukiwanie hiperparametrów: kompromis czasu/eksploracji
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=8,  # ograniczona liczba prób
                cv=3,  # walidacja krzyżowa
                scoring="f1_macro",  # średnia po klasach
                n_jobs=1,  # procesy sekwencyjnie
                random_state=42, # standardowa wartość
                verbose=0, # szczegóły
                error_score="raise"  # wyjątek
            )

            start_time = time.perf_counter()
            try:
                search.fit(X_train, y_train)  # trenowanie + CV
                y_pred = search.predict(X_test)
                end_time = time.perf_counter()
                elapsed = end_time - start_time

                # Zbieranie wyników do raportów
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                results[model_name] = {
                    "best_params": search.best_params_,
                    "report_df": pd.DataFrame(report).T,
                    "confusion_df": pd.DataFrame(cm),
                    "time_sec": elapsed
                }
                print(f"    Najlepsze parametry: {search.best_params_}")
                print(f"    Czas: {elapsed:.1f}s") # Podczas mojej nieobecności wiedzieć ile czasu zajmowało
            except Exception as e:
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                print(f"    Błąd przy trenowaniu {model_name} na {ds_name}: {e}")
                # Zapisuje błąd żeby w raporcie było widać co zawiodło
                results[model_name] = {"error": str(e), "time_sec": elapsed}

        # Zapis wszystkich wyników do Excela
        out_path = f"results/r_{ds_name}_best_model.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for model_name, content in results.items():
                if "error" in content:
                    # osobny arkusz z błędem jak coś pójdzie nie tak
                    pd.DataFrame({"error": [content["error"]]}).to_excel(writer, sheet_name=f"{model_name}_error")
                else:
                    content["report_df"].to_excel(writer, sheet_name=f"{model_name}_metrics")
                    content["confusion_df"].to_excel(writer, sheet_name=f"{model_name}_confusion")
                    pd.DataFrame.from_dict({k: [v] for k, v in content["best_params"].items()}).to_excel(
                        writer, sheet_name=f"{model_name}_params"
                    )
                    pd.DataFrame({"time_seconds": [content.get("time_sec", None)]}).to_excel(
                        writer, sheet_name=f"{model_name}_time"
                    )
        print(f"Ścieżka zapisu raportu: {out_path}")


if __name__ == "__main__":
    run_all()