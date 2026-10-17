"""Datenaufbereitung, Topic-Zuordnung und Vergleich der ausgewählten Verfahren.

Die Funktionen erhalten ihre Daten und Einstellungen als Argumente.
Ein Import lädt keine CSV und startet kein Modelltraining.
"""
from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
from gensim.models import CoherenceModel
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score,
)

def prepare_records(
    source_df: pd.DataFrame,
    text_column: str,
    min_words: int,
    id_column: str,
):
    """Doppelte Datensätze entfernen und Meldungen mit unterschiedlichen IDs beibehalten."""
    if not source_df.columns.is_unique:
        raise ValueError("Die Spaltennamen müssen eindeutig sein.")
    missing_columns = [col for col in (text_column, id_column) if col not in source_df.columns]
    if missing_columns:
        raise ValueError(f"Erforderliche Spalten fehlen: {missing_columns}")
    if text_column == id_column:
        raise ValueError("Textspalte und ID-Spalte müssen unterschiedlich sein.")
    if not isinstance(min_words, int) or isinstance(min_words, bool) or min_words < 1:
        raise ValueError("min_words muss eine ganze Zahl größer oder gleich 1 sein.")

    counts = {"input": len(source_df)}

    # Fehlende IDs nicht als gemeinsame Meldung behandeln.
    id_keys = source_df[id_column].astype("string").str.strip()
    if id_keys.isna().any() or id_keys.eq("").any():
        raise ValueError("Meldungs-IDs fehlen. Die betroffenen Datensätze müssen geprüft werden.")

    # Nur über alle Spalten identische Datensätze entfernen, einschließlich der ID.
    work = source_df.drop_duplicates(keep="first").copy()
    counts["duplicates"] = len(source_df) - len(work)

    # Bei unterschiedlichen Inhalten zur selben ID keinen Datensatz willkürlich auswählen.
    if work[id_column].astype("string").str.strip().duplicated(keep=False).any():
        raise ValueError(
            "Zu einer Meldungs-ID liegen unterschiedliche Datensätze vor. "
            "Die Unterschiede müssen vor der Analyse geprüft werden."
        )

    before = len(work)
    work = work.dropna(subset=[text_column]).copy()
    counts["missing"] = before - len(work)
    work[text_column] = work[text_column].astype(str).str.strip()

    before = len(work)
    work = work.loc[work[text_column].str.len() > 0].copy()
    counts["empty"] = before - len(work)

    # Gleiche Texte mit unterschiedlichen IDs zählen weiterhin als einzelne Meldungen.
    work["word_count_raw"] = work[text_column].str.split().str.len()
    before = len(work)
    work = work.loc[work["word_count_raw"] >= min_words].copy()
    counts["raw_short"] = before - len(work)
    counts["remaining"] = len(work)
    counts["text_repetitions"] = int(work[text_column].duplicated().sum())

    if counts["input"] != sum(counts[key] for key in (
        "duplicates", "missing", "empty", "raw_short", "remaining"
    )):
        raise RuntimeError("Die dokumentierten Zeilenzahlen sind nicht konsistent.")

    return work.reset_index(drop=True), counts

def is_location_token(token: str) -> bool:
    # Mögliche Straßennamen anhand der Endung erkennen und "straße" ausnehmen.
    if token == "straße":
        return False

    if token.endswith(("strasse", "straße", "str")) and len(token) > 5:
        return True

    # Mögliche Wegnamen erkennen und die folgenden allgemeinen Wegbegriffe ausnehmen.
    if token in {"gehweg", "radweg", "fahrradweg"}:
        return False

    if token.endswith("weg") and len(token) > 5:
        return True

    return False

def normalization_key(word: str) -> str:
    """Umlautschreibweisen nur für den Abgleich bekannter Wörter vereinheitlichen."""
    return word.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")

def build_normalization_lookup(base: dict, groups: dict) -> dict:
    """Bekannte Schreibweisen zusammenführen und verkettete Zuordnungen auflösen."""
    lookup = {}
    for source, target in base.items():
        key = normalization_key(source)
        if key in lookup and normalization_key(lookup[key]) != normalization_key(target):
            raise ValueError(f"Widersprüchliche Begriffszuordnung für {source}.")
        lookup[key] = target
    for target, forms in groups.items():
        for form in forms:
            lookup[normalization_key(form)] = target
    resolved = {}
    for source in lookup:
        key, seen = source, set()
        while key in lookup:
            if key in seen:
                raise ValueError(f"Zyklische Begriffszuordnung für {source}.")
            seen.add(key)
            target = lookup[key]
            next_key = normalization_key(target)
            if next_key == key or next_key not in lookup:
                resolved[source] = target
                break
            key = next_key
    return resolved

def clean_text(text: str, *, lookup: dict, stopwords: set, min_token_length: int,
               phrases: tuple[str, ...]) -> str:
    """Die im Variantenvergleich geprüften Textregeln anwenden."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    for expression in phrases:
        text = re.sub(expression, " ", text)
    text = re.sub(r"[^a-zäöüß\s]", " ", text)
    tokens = []
    for original in text.split():
        token = lookup.get(normalization_key(original), original)
        if is_location_token(token):
            continue
        if normalization_key(token) not in stopwords and len(token) >= min_token_length:
            tokens.append(token)
    return " ".join(tokens)

def select_model_documents(documents, count_matrix, tfidf_matrix):
    """Die gleiche Auswahl von Meldungen auf beide Merkmalsmatrizen anwenden."""
    if len(documents) != count_matrix.shape[0] or count_matrix.shape != tfidf_matrix.shape:
        raise ValueError("Meldungen und Merkmalsmatrizen passen nicht zusammen.")
    if not documents.index.is_unique:
        raise ValueError("Der Index der Meldungen muss eindeutig sein.")

    count_matrix = count_matrix.tocsr(copy=True)
    tfidf_matrix = tfidf_matrix.tocsr(copy=True)
    for matrix in (count_matrix, tfidf_matrix):
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("Merkmalsmatrizen müssen endliche, nichtnegative Werte enthalten.")
        matrix.eliminate_zeros()

    count_mask = np.asarray(count_matrix.getnnz(axis=1)).ravel() > 0
    tfidf_mask = np.asarray(tfidf_matrix.getnnz(axis=1)).ravel() > 0
    if not np.array_equal(count_mask, tfidf_mask):
        raise ValueError("Count und TF-IDF ergeben unterschiedliche Meldungen ohne Merkmale.")

    # Die ursprünglichen Zeilenpositionen bleiben über den Index nachvollziehbar.
    fit_documents = documents.iloc[np.flatnonzero(count_mask)].copy()
    without_features = documents.iloc[np.flatnonzero(~count_mask)].copy()
    return (
        count_mask,
        fit_documents,
        count_matrix[count_mask],
        tfidf_matrix[count_mask],
        without_features,
    )

def normalize_nmf_for_assignment(coefficients, components):
    """Topic-Vektoren auf L2-Norm 1 bringen und die Koeffizienten entsprechend anpassen."""
    w = np.asarray(coefficients, dtype=float)
    h = np.asarray(components, dtype=float)
    if w.ndim != 2 or h.ndim != 2 or w.shape[1] != h.shape[0] or h.shape[0] == 0:
        raise ValueError("Die beiden NMF-Matrizen passen nicht zusammen.")
    if h.shape[1] == 0:
        raise ValueError("Die Topic-Vektoren enthalten keine Merkmale.")
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (w, h)):
        raise ValueError("NMF-Koeffizienten müssen endlich und nicht negativ sein.")

    lengths = np.linalg.norm(h, axis=1)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Mindestens ein Topic-Vektor hat keine verwendbare L2-Norm.")
    h_unit = h / lengths[:, None]
    w_scaled = w * lengths[None, :]
    if not np.isfinite(w_scaled).all():
        raise ValueError("Die angepassten NMF-Koeffizienten enthalten ungültige Werte.")
    if not np.allclose(np.linalg.norm(h_unit, axis=1), 1.0):
        raise RuntimeError("Die Topic-Vektoren haben nicht die erwartete L2-Norm.")

    # Das Produkt beider Matrizen muss erhalten bleiben, kleine Abweichungen in der Rundung sind zulässig.
    for start in range(0, len(w), 256):
        before = w[start:start + 256] @ h
        after = w_scaled[start:start + 256] @ h_unit
        if not np.allclose(before, after, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Die Rekonstruktion wurde durch die Skalierung verändert.")
    return w_scaled, h_unit

def assign_topics(doc_topic_matrix, model_mask, source_index):
    """Modellgewichte den ursprünglichen Zeilen zuordnen und fehlende Merkmale kennzeichnen."""
    weights = np.asarray(doc_topic_matrix, dtype=float)
    mask = np.asarray(model_mask)
    if mask.ndim != 1 or mask.dtype != bool or len(mask) != len(source_index):
        raise ValueError("Die Zeilenauswahl passt nicht zum Index der Meldungen.")
    if not source_index.is_unique:
        raise ValueError("Der Index der Meldungen muss eindeutig sein.")
    if weights.ndim != 2 or weights.shape[0] != int(mask.sum()) or weights.shape[1] == 0:
        raise ValueError("Die Dokument-Topic-Matrix passt nicht zur Zeilenauswahl.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Topic-Gewichte müssen endlich und nichtnegativ sein.")

    result = pd.DataFrame(index=source_index.copy())
    result["Topic"] = pd.Series(pd.NA, index=source_index, dtype="Int64")
    result["Status"] = pd.Series(
        "Keine Merkmale im Vokabular", index=source_index, dtype="string"
    )
    result["Gleicher Höchstwert"] = False
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return result

    max_weights = weights.max(axis=1)
    positive = max_weights > 0
    valid_index = source_index.take(positions[positive])
    zero_index = source_index.take(positions[~positive])

    # Nur positive Gewichte für eine Zuordnung verwenden.
    # Bei exakt gleichen Höchstwerten wählt argmax die erste Topic-Nummer.
    labels = weights[positive].argmax(axis=1) + 1
    result.loc[valid_index, "Topic"] = labels
    result.loc[valid_index, "Status"] = "Topic zugeordnet"
    result.loc[zero_index, "Status"] = "Keine positiven Topic-Gewichte"
    tied = (weights[positive] == max_weights[positive, None]).sum(axis=1) > 1
    result.loc[valid_index, "Gleicher Höchstwert"] = tied
    return result

def topic_dominance(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Anteil des stärksten Werts berechnen, ohne daraus eine Trefferwahrscheinlichkeit abzuleiten."""
    w = np.asarray(coefficients, dtype=float)
    if w.ndim != 2 or w.shape[1] < 2 or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Die Dokument-Topic-Matrix muss endlich, nicht negativ und zweidimensional sein.")
    highest = w.max(axis=1)
    positive = highest > 0
    dominance = np.full(len(w), np.nan)
    # Diese rechnerisch gleichwertige Form vermeidet einen Überlauf bei großen Zeilensummen.
    dominance[positive] = 1.0 / (w[positive] / highest[positive, None]).sum(axis=1)
    ties = positive & ((w == highest[:, None]).sum(axis=1) > 1)
    return dominance, ties

def describe_comparison_run(model, coefficients, method: str, vectorization: str,
                            *, documents, fit_index, mask, features,
                            coherence_texts, coherence_dictionary, top_n, window_size,
                            topic_coherence=None, messages=()) -> dict:
    """Topic-Begriffe und Zuordnungen eines Laufs getrennt von der Hauptanalyse erfassen."""
    h = np.asarray(model.components_, dtype=float)
    if (h.shape != (model.n_components, len(features))
            or not np.isfinite(h).all() or (h < 0).any()
            or ((h > 0).sum(axis=1) < top_n).any()):
        raise ValueError("Die Komponenten passen nicht zum gemeinsamen Vokabular.")
    words = [features[idx].tolist() for idx in
             np.argsort(-h, axis=1, kind="stable")[:, :top_n]]
    if topic_coherence is None:
        topic_coherence = CoherenceModel(
            topics=words, texts=coherence_texts, dictionary=coherence_dictionary,
            coherence="c_v", topn=top_n,
            window_size=window_size, processes=1,
        ).get_coherence_per_topic()
    cv = np.asarray(topic_coherence, dtype=float)
    if cv.shape != (model.n_components,) or not np.isfinite(cv).all():
        raise ValueError("Die Kohärenz ist nicht für alle Topics berechenbar.")
    assignments = assign_topics(coefficients, mask, documents.index)
    dominance, _ = topic_dominance(coefficients)
    assignments["Dominanz"] = pd.Series(dominance, index=fit_index).reindex(documents.index)
    counts = assignments["Topic"].value_counts().reindex(range(1, model.n_components + 1), fill_value=0)
    topics = pd.DataFrame({
        "Verfahren": method, "Vektorisierung": vectorization, "Seed": model.random_state,
        "Topic": range(1, model.n_components + 1),
        "Top-Begriffe": [", ".join(row) for row in words], "c_v": cv,
        "Anzahl Meldungen": counts.to_numpy(dtype=int),
        "Anteil Prozent": 100 * counts.to_numpy() / len(documents),
    })
    return {"model": model, "coefficients": coefficients, "assignments": assignments,
            "topics": topics, "coherence": cv, "warnings": list(messages)}

def fit_comparison_run(model, matrix, method: str, vectorization: str, **context) -> dict:
    """Einen Vergleich berechnen, ohne das ausgewählte NMF-Modell zu verändern."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        coefficients = model.fit_transform(matrix.copy())
    messages = [str(item.message) for item in recorded]
    for message in messages:
        print("Warnung:", message)
    if method == "NMF":
        coefficients, _ = normalize_nmf_for_assignment(coefficients, model.components_)
    elif not np.allclose(coefficients.sum(axis=1), 1.0):
        raise RuntimeError("Die LDA-Themenanteile summieren sich nicht zu eins.")
    return describe_comparison_run(model, coefficients, method, vectorization, messages=messages, **context)

def comparison_category_scores(labels: pd.Series, categories: pd.Series) -> dict:
    """Übereinstimmung mit vorhandenen Kategorien auf derselben Auswahl berechnen."""
    categories = categories.astype("string").str.strip().replace("", pd.NA)
    if not labels.index.equals(categories.index) or labels.isna().any():
        raise ValueError("Topic-Zuordnungen und Kategorien müssen zeilengleich vorliegen.")
    valid = categories.notna()
    n = int(valid.sum())
    result = {"Bewertete Meldungen": n, "Ohne Kategorie": int((~valid).sum()),
              "NMI": np.nan, "AMI": np.nan, "ARI": np.nan, "Reinheit": np.nan}
    if n == 0:
        return result
    truth, topics = categories.loc[valid].astype(str), labels.loc[valid].astype(int)
    table = pd.crosstab(topics, truth)
    result["Reinheit"] = float(table.max(axis=1).sum() / n)
    if n >= 2:
        result.update({
            "NMI": normalized_mutual_info_score(truth, topics, average_method="arithmetic"),
            "AMI": adjusted_mutual_info_score(truth, topics, average_method="arithmetic"),
            "ARI": adjusted_rand_score(truth, topics),
        })
    return result

def summarize_comparison_runs(method_runs: dict, documents: pd.DataFrame,
                              features: np.ndarray, id_column: str):
    """Kennzahlen je Lauf und deren Mittelwerte auf gemeinsamer Datenbasis berechnen."""
    comparison_common = pd.Series(True, index=documents.index)
    for run in method_runs.values():
        comparison_common &= run["assignments"]["Topic"].notna()
    if not comparison_common.any():
        raise ValueError("Die Modelle besitzen keine gemeinsam zugeordneten Meldungen.")

    comparison_rows, comparison_assignments = [], []
    for (method, vectorization, seed), run in method_runs.items():
        assignments, model = run["assignments"], run["model"]
        row = {
            "Verfahren": method, "Vektorisierung": vectorization, "Seed": seed,
            "Anzahl Topics": model.n_components, "max_iter": model.max_iter,
            "Iterationen": model.n_iter_, "Nach Text Preprocessing": len(documents),
            "Merkmale": len(features), "Zugeordnet": int(assignments["Topic"].notna().sum()),
            "Nicht zugeordnet": int(assignments["Topic"].isna().sum()),
            "c_v Mittelwert": float(run["coherence"].mean()),
            "c_v Minimum": float(run["coherence"].min()),
            "Gemeinsam zugeordnet": int(comparison_common.sum()),
            "Hier neu trainiert": not (method == "NMF" and vectorization == "TF-IDF"),
            "Warnungen neuer Lauf": len(run["warnings"]), "Hinweise": " | ".join(run["warnings"]),
        }
        for column, suffix in (("category_parent", "Haupt"), ("category_child", "Unter")):
            scores = comparison_category_scores(
                assignments.loc[comparison_common, "Topic"], documents.loc[comparison_common, column],
            )
            row.update({f"{name} {suffix}": value for name, value in scores.items()})
        comparison_rows.append(row)
        export = assignments.copy()
        export.insert(0, "ID", documents[id_column].astype("string").str.strip().to_numpy())
        export.insert(0, "Seed", seed)
        export.insert(0, "Vektorisierung", vectorization)
        export.insert(0, "Verfahren", method)
        export["Gemeinsam zugeordnet"] = comparison_common
        comparison_assignments.append(export.reset_index(drop=True))

    method_comparison_runs = pd.DataFrame(comparison_rows)
    method_comparison_topics = pd.concat([r["topics"] for r in method_runs.values()], ignore_index=True)
    method_comparison_assignments = pd.concat(comparison_assignments, ignore_index=True)
    metrics = ["c_v Mittelwert"] + [f"{name} {level}" for level in ("Haupt", "Unter")
                                    for name in ("NMI", "AMI", "ARI", "Reinheit")]
    summary_rows, display_rows = [], []
    for (method, vectorization), part in method_comparison_runs.groupby(["Verfahren", "Vektorisierung"], sort=False):
        row = {"Verfahren": method, "Vektorisierung": vectorization,
               "Anzahl Topics": int(part["Anzahl Topics"].iloc[0]), "Läufe": len(part),
               "Gemeinsam zugeordnet": int(comparison_common.sum())}
        shown = {"Konfiguration": f"{method} mit {vectorization}", "Topics": row["Anzahl Topics"], "Läufe": len(part)}
        for metric in metrics:
            mean, std = float(part[metric].mean()), float(part[metric].std(ddof=1))
            row[f"{metric} Mittel"] = mean
            row[f"{metric} Standardabweichung"] = std
            if metric in ("c_v Mittelwert", "AMI Haupt", "AMI Unter", "Reinheit Haupt"):
                shown[metric] = f"{mean:.4f}" + (f" ± {std:.4f}" if len(part) > 1 else "")
        summary_rows.append(row)
        display_rows.append(shown)
    method_comparison_summary = pd.DataFrame(summary_rows)

    if len(method_comparison_runs) != 7 or method_comparison_assignments.duplicated(
        ["Verfahren", "Vektorisierung", "Seed", "ID"]
    ).any():
        raise RuntimeError("Der Vergleich enthält nicht die vorgesehenen sieben eindeutigen Läufe.")
    return (method_comparison_runs, method_comparison_topics,
            method_comparison_assignments, method_comparison_summary,
            comparison_common, pd.DataFrame(display_rows))
