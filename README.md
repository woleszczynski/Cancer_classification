# Porównanie efektywności wybranych modeli uczenia maszynowego w rozpoznawaniu typów nowotworów

## 🎯 Cel projektu
Celem projektu jest opracowanie i porównanie skuteczności różnych metod uczenia maszynowego w zadaniu klasyfikacji typów nowotworów, takich jak rak piersi, płuc, jelita grubego i żołądka. Analizowane są zarówno klasyczne modele ML (SVM, drzewa decyzyjne), jak i zaawansowane techniki zespołowe (Random Forest, Gradient Boosting).

## 📚 Opis pracy
Projekt obejmuje:
- wstępne przetwarzanie danych medycznych (usuwanie braków, normalizacja, selekcja cech, równoważenie klas),
- implementację i trening wybranych modeli przy użyciu scikit-learn,
- ocenę skuteczności klasyfikacji przy użyciu metryk: Accuracy, Precision, Recall, F1-score.

Ostatecznym celem jest identyfikacja najbardziej efektywnych metod klasyfikacji nowotworów oraz wnioski dotyczące ich zastosowania w diagnostyce medycznej.

## 📂 Struktura repozytorium
project-root/
├── results/ # wyniki eksperymentów w formacie Excel
│ ├── best_model
│ 	├── r_SEER_best_model.xlsx
│ 	├── r_Breast_best_model.xlsx
│ 	├── r_TCGA_best_model.xlsx
│ 	└── r_Lung_best_model.xlsx
│ ├── final
│ 	├── r_SEER_final.xlsx
│ 	├── r_Breast_final.xlsx
│ 	├── r_TCGA_final.xlsx
│ 	└── r_Lung_final.xlsx
├── src/ # kod źródłowy
│ ├── best_models.py
│ └── final.py
└── README.md


## 📊 Wyniki

W folderze results/ znajdują się pliki Excel zawierające:

- raporty metryk (*_metrics),

- macierze pomyłek (*_confusion),

- najlepsze parametry modeli (*_params),

- czasy wykonania (*_time).


## ⚙️ Modele i metryki

Modele: Decision Tree, Random Forest, SVM, Gradient Boosting / HistGradientBoosting

Metryki: Accuracy, Precision, Recall, F1-score


## 🔗 Pipeline
raw data → preprocessing → best_models.py → final.py → results


## ℹ️ Uwagi

Pliki danych nie są dołączone do repozytorium ze względu na rozmiar.
