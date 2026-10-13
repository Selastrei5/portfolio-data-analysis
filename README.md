# Themenanalyse kommunaler Bürgerhinweise

Python-Projekt zur Analyse von unstrukturierten Beschwerden und Hinweisen von Bürgern aus dem Mängelmelder der Stadt Konstanz. Der Code liegt als Jupyter Notebook vor und verwendet scikit-learn für die Modellierung der Themen mit LDA und NMF. Die Ausgabe umfasst eine nach Häufigkeit sortierte Themenübersicht und ein Balkendiagramm mit matplotlib.

## Funktionen

| Funktion | Umsetzung |
|---|---|
| **Data Profiling** | Analyse von Fehlwerten, Textduplikaten und der Verteilung der Textlängen mit `pandas` |
| **Text Preprocessing** | Bereinigung mit regulären Ausdrücken, Normalisierung von Schreibvarianten und Begriffen, Tokenisierung und Stopwortfilterung mit `re` und `pandas` |
| **Vektorisierung und Themenmodellierung** | `CountVectorizer` mit LDA sowie `TfidfVectorizer` mit NMF in `scikit-learn` |
| **Vergleich der Stopwörter** | Vergleich von drei Varianten zur Filterung von Stopwörtern mit unterschiedlicher Behandlung von Zustandsbeschreibungen und Negationen, jeweils für LDA und NMF bei zehn Topics |
| **Modellevaluation** | Kohärenzbewertung mit `c_v` in `gensim` und qualitative Prüfung ausgewählter Meldungen |
| **Topic-Auswertung und Visualisierung** | Zuordnung zum dominanten Topic, Berechnung absoluter und relativer Häufigkeiten, CSV-Export und Balkendiagramm mit `NumPy`, `pandas` und `matplotlib` |

## Daten

Der Code ist auf den öffentlich verfügbaren Datensatz des Mängelmelders der Stadt Konstanz abgestimmt, die CSV-Datei wird an folgendem Pfad erwartet:

```text
data/raw/maengelmelder_konstanz_final_9_neu.csv
```

Der Code analysiert die Meldetexte aus der Spalte `text_raw`, während die vorhandenen Haupt- und Unterkategorien ausschließlich zur Plausibilisierung der Ergebnisse dienen. Die Rohdaten sind nicht Bestandteil des Repositorys. Andere Datensätze erfordern Anpassungen an Spalten, Vorverarbeitung und Themenbezeichnungen.

## Nutzung

Die Entwicklung erfolgt mit Python 3.11 und Conda unter Windows. `environment.yml` dokumentiert den verwendeten Paketstand einschließlich Windows-spezifischer Abhängigkeiten, der Neuaufbau der Umgebung ist noch nicht getestet.

Zum Start mit der bereits eingerichteten Umgebung sollten die folgenden Befehle im Repository-Ordner ausführt werden:

```bat
conda activate portfolio_da
jupyter notebook
```

In Jupyter `notebooks/01_poc.ipynb` öffnen, den Kernel `Python (portfolio_da)` auswählen und die Zellen von oben nach unten ausführen.

| Notebook | Inhalt |
|---|---|
| [01_poc.ipynb](notebooks/01_poc.ipynb) | Aktuelle Analyse, Modellvergleich und Ergebnisdarstellung |
| [02_bisherige_vergleiche.ipynb](notebooks/02_bisherige_vergleiche.ipynb) | Frühere Vergleiche mit den damaligen Einstellungen |

Die aktuelle Analyse lässt sich unabhängig vom zweiten Notebook ausführen, es dokumentiert nur den Vergleich.

## Modellvergleich

Für LDA und NMF wurden jeweils 6, 8, 10, 12 und 15 Topics bei unverändertem Text Preprocessing verglichen. Die folgende Tabelle zeigt die mittleren Kohärenzwerte (`c_v`).

| Anzahl der Topics | LDA | NMF |
|---:|---:|---:|
| 6 | 0,3430 | 0,3532 |
| 8 | 0,3096 | 0,3922 |
| 10 | **0,3705** | 0,3844 |
| 12 | 0,3662 | **0,4201** |
| 15 | 0,3632 | 0,4120 |

Für die Ergebnisübersichten wurden LDA mit zehn und NMF mit zwölf Topics ausgewählt, die Ergebnisse wurden ergänzend anahnd ausgewählter Meldungen inhaltlich geprüft. Die Darstellung im Diagramm basiert auf NMF, bei der NMF-Zuordnung werden die Topic-Vektoren L2-normiert und die Koeffizienten entsprechend angepasst.

## Ergebnisse

Die Dateien der Analyse entstehen lokal unter `outputs/`. Das zweite Notebook speichert seine Ergebnisse getrennt unter `outputs/bisherige_vergleiche/`.

| Ausgabe | Inhalt |
|---|---|
| `themenuebersicht_lda.csv`, `themenuebersicht_nmf.csv` | Themenübersichten mit Häufigkeiten und wichtigen Begriffen |
| `vergleich_topic_anzahl.csv`, `vergleich_topic_anzahl_details.csv` | Kohärenzwerte und Häufigkeiten je Modell und Topic |
| `modellauswahl.csv` | Ausgewählte Modelle und zugehörige Kennzahlen |
| `nmf_topics_beschriftet.csv` | Alle zwölf NMF-Topics mit Bezeichnungen, Beschreibungen und Einordnung |
| `entscheidungstraeger_themenuebersicht.csv` | Sieben benannte Schwerpunkte mit Rang, Häufigkeit, Anteil, Kurzbeschreibung und beispielhaften Hinweisen |
| `umfang_themenuebersicht.csv` | Anzahl und Anteil der dargestellten und nach Themen, die nicht aufgeschlüsselten werden konnten |
| `zuordnung_lda_nmf.csv` | Anzahl und Anteil der zugeordneten und nicht zugeordneten Meldungen für beide Verfahren |
| `themenhaeufigkeiten_nmf.png` | Horizontales Balkendiagramm der sieben Schwerpunkte |

Die kommunale Übersicht zeigt 1.658 von 2.759 Meldungen (60,1 %). Weitere 1.101 Meldungen (39,9 %) werden nicht nach Themen aufgeschlüsselt. 

## Hinweise zur Interpretation

Die Rangfolge zeigt die Häufigkeiten der dargestellten Schwerpunkte und nicht die Dringlichkeit der Probleme in den Meldungen. Die Prozentwerte beziehen sich auf alle Meldungen nach dem Preprocessing. Die Häufigkeiten beruhen auf der automatischen Zuordnung, die Themenbezeichnungen auf der Prüfung ausgewählter Meldungen. Etwaige Meldungen ohne Merkmale erhalten keine Topic-Zuordnung und werden separat ausgewiesen.

## Entwicklungsstand

Folgende Prüfungen und Ergänzungen stehen noch aus.

- **Modellqualität** anhand derselben Meldungen für LDA und NMF weiter prüfen.
- **Datenquelle und Bereitstellung** der benötigten CSV dokumentieren.
- **Installation und Nutzung** in einer neu erstellten Python-Umgebung testen.
