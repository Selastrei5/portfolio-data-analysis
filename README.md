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

## Ergänzende Modellvergleiche (Schritte 11g–11l)

Nach der ersten Modellauswahl wurden Textverarbeitung, Anzahl der Topics, Vektorisierung und Trainingseinstellungen gezielt verglichen. Zur Bewertung dienten die Kohärenz `c_v`, die Übereinstimmung mit den vorhandenen Meldekategorien (NMI, AMI, ARI und Reinheit) und die inhaltliche Prüfung ausgewählter Meldungen. Die direkten Kategorienvergleiche beziehen sich auf dieselben 2.651 Meldungen mit Topic-Zuordnung.

| Schritt | Untersuchung | Ergebnis |
|---|---|---|
| 11g | Gemeinsame Stichprobe | LDA und NMF wurden an denselben 100 Meldungen gegenübergestellt. Die KI-gestützte Erstbewertung zeigte gemischte Topics und unpassende Zuordnungen. |
| 11h | Textnormalisierung und Vokabularfilter | Die Meldetexte verwenden ae, oe und ue statt Umlauten. Die Normalisierung von Schreibweisen und Wortformen zusammen mit `min_df=2` reduziert die Anzahl der Meldungen ohne Merkmale von 108 auf 28. Das Absenken von `min_df` allein bringt keinen durchgängigen inhaltlichen Vorteil. |
| 11i | Vergleich mit den Meldekategorien und Prüfung der Topic-Dominanz | Für NMF mit zwölf Topics steigt die NMI gegenüber den Hauptkategorien von 0,316 in der Referenz auf 0,333 bei der Kombination aus Normalisierung und `min_df=2`. LDA mit zehn Topics erreicht bei derselben Kombination 0,196. Die Dominanz des stärksten Topics wurde anhand der Schwellenwerte 0,4, 0,5 und 0,6 untersucht, ohne die bisherige Zuordnung oder Zählung der Meldungen zu ändern. |
| 11j | Anzahl der NMF-Topics | Je Textvariante wurden 6, 8, 10, 12 und 15 Topics verglichen. Bei der Kombination erreicht NMF mit zehn Topics `c_v=0,447`, mit fünfzehn Topics `c_v=0,438`. Mit fünfzehn Topics entstehen zusätzliche Gruppen, unter anderem für Sperrmüll und abgemeldete Fahrzeuge. |
| 11k | Count, TF-IDF und Initialisierung | Bei NMF mit fünfzehn Topics erreicht Count höhere Kohärenzwerte als TF-IDF (`c_v` im Mittel 0,545 gegenüber 0,438), enthält aber auch einzelne schwer interpretierbare Topics aus allgemeinen Formulierungen. Beide Varianten liefern über fünf Seeds stabile Gruppierungen. |
| 11l | LDA-Trainingsumfang und Initialisierung | 100 statt 20 Iterationen verbessern die Kennzahlen nur geringfügig. Die Gruppierungen unterscheiden sich weiterhin deutlich zwischen den Seeds. Die paarweisen ARI-Werte liegen über die untersuchten Textvarianten und Trainingslimits hinweg zwischen 0,193 und 0,376. |

Für die Ergebnisübersicht wurde **NMF mit TF-IDF, fünfzehn Topics, der untersuchten Textnormalisierung und `min_df=2`** ausgewählt.

Count erreichte zwar bei mehreren Kennzahlen höhere Werte, mit TF-IDF ließen sich in der inhaltlichen Prüfung jedoch mehrere Themen klarer erkennen. Für den Vergleich wurden die Anliegen aus 50 Meldungen manuell zusammengefasst und die Zuordnungen beider Modelle anschließend mithilfe von KI beurteilt. Count und LDA bleiben als Vergleichsverfahren dokumentiert.

Die Übernahme in die Hauptanalyse und die neue Beschriftung der Themen folgen im nächsten Schritt. Bis dahin beziehen sich die bisherigen Ausgaben der Ergebnisse und das Diagramm weiterhin auf die Referenz mit zwölf NMF-Topics.

## Hinweise zur Interpretation

Die Rangfolge zeigt die Häufigkeiten der dargestellten Schwerpunkte und nicht die Dringlichkeit der Probleme in den Meldungen. Die Prozentwerte beziehen sich auf alle Meldungen nach dem Preprocessing. Die Häufigkeiten beruhen auf der automatischen Zuordnung, die Themenbezeichnungen auf der Prüfung ausgewählter Meldungen. Etwaige Meldungen ohne Merkmale erhalten keine Topic-Zuordnung und werden separat ausgewiesen.

## Entwicklungsstand

Folgende Prüfungen und Ergänzungen stehen noch aus.

- **Modellqualität** anhand derselben Meldungen für LDA und NMF weiter prüfen.
- **Datenquelle und Bereitstellung** der benötigten CSV dokumentieren.
- **Installation und Nutzung** in einer neu erstellten Python-Umgebung testen.
