# Themenanalyse kommunaler Bürgerhinweise

Python-Projekt zur Analyse von unstrukturierten Beschwerden und Hinweisen von Bürgern aus dem Mängelmelder der Stadt Konstanz. Der Code liegt als Jupyter Notebook vor und verwendet scikit-learn für die Themenmodellierung mit LDA und NMF. Die Ausgabe umfasst eine nach Häufigkeit sortierte Themenübersicht und ein Balkendiagramm mit matplotlib.

## Funktionen

| Funktion | Umsetzung |
|---|---|
| **Data Profiling** | Analyse von Fehlwerten, Textduplikaten und der Verteilung der Textlängen mit `pandas` |
| **Text Preprocessing** | Bereinigung mit regulären Ausdrücken, Normalisierung von Schreibvarianten und Begriffen, Tokenisierung und Stopwortfilterung mit `re` und `pandas` |
| **Vektorisierung und Themenmodellierung** | `CountVectorizer` mit LDA sowie `TfidfVectorizer` mit NMF in `scikit-learn` |
| **Topic-Auswertung und Visualisierung** | Zuordnung zum dominanten Topic, Berechnung absoluter und relativer Häufigkeiten, CSV-Export und Balkendiagramm mit `NumPy`, `pandas` und `matplotlib` |

## Daten

Der Code ist auf den öffentlich verfügbaren Datensatz des Mängelmelders der Stadt Konstanz abgestimmt. Die CSV-Datei wird an diesem Pfad erwartet.

```text
data/raw/maengelmelder_konstanz_final_9_neu.csv
```

Der Code analysiert die Meldetexte aus der Spalte `text_raw`, während die vorhandenen Haupt- und Unterkategorien ausschließlich zur Plausibilisierung der Ergebnisse dienen. Die Rohdaten sind nicht Bestandteil des Repositorys. Andere Datensätze erfordern Anpassungen an Spalten, Vorverarbeitung und Themenbezeichnungen.

## Nutzung

Die Entwicklung erfolgt mit Python 3.11 und Conda unter Windows. `environment.yml` dokumentiert den bisherigen Paketstand einschließlich Windows-spezifischer Abhängigkeiten. Der Neuaufbau der Umgebung ist noch nicht getestet.

Zum Start mit der bereits eingerichteten Umgebung die folgenden Befehle im Repository-Ordner ausführen.

```bat
conda activate portfolio_da
jupyter notebook
```

In Jupyter `notebooks/01_poc.ipynb` öffnen, den Kernel `Python (portfolio_da)` auswählen und die Zellen von oben nach unten ausführen.

## Ergebnisse

Die Dateien entstehen lokal unter `outputs/`.

| Ausgabe | Inhalt |
|---|---|
| CSV-Dateien für LDA und NMF | Technische Themenübersichten mit Häufigkeiten und wichtigen Begriffen |
| CSV-Datei für Entscheidungstragende | NMF-Themenübersicht mit Rang, Beschwerdethema, Kurzbeschreibung, Anzahl, Anteil und beispielhaften Hinweisen |
| PNG-Datei | Horizontales Balkendiagramm der Häufigkeiten je Thema |
| `zuordnung_lda_nmf.csv` | Anzahl und Anteil der zugeordneten und nicht zugeordneten Meldungen für beide Verfahren |

Die Themenübersicht für Entscheidungstragende und das Balkendiagramm sind zunächst deaktiviert, bis die Themenbezeichnungen für den neuen Modellstand geprüft sind.



## Hinweise zur Interpretation

Die Rangfolge zeigt die häufigsten Themen und nicht die dringendsten Probleme. Die Prozentwerte beziehen sich auf alle Meldungen nach dem Preprocessing. Meldungen ohne Merkmale erhalten keine Topic-Zuordnung und werden separat ausgewiesen.

## Entwicklungsstand

Der aktuelle Stand ist ein Prototyp. Folgende Erweiterungen und Korrekturen sind geplant:


- **Text Preprocessing** muss hinsichtlich der Auswahl der Stopwörter und der Behandlung identischer Meldetexte geprüft werden.
- **Modellvergleich** mit unterschiedlicher Anzahl von Topics für LDA und NMF. Die Auswahl berücksichtigt Kohärenzwerte (`c_v`) und die inhaltliche Interpretierbarkeit.
- **Themenbezeichnungen und Ergebnisdarstellung** an den ausgewählten Modellstand anpassen.
- **Installation und Nutzung** in einer neu erstellten Python-Umgebung testen.
