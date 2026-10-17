"""Beschriftungen, Ergebnistabellen und Diagramm der ausgewählten Analyse."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

LABEL_DATA_SHA256 = "bd7057df23579df4302fea4a51302c9f6459a1624cee55e47cf331e117eae84c"
LABELS_VERSION = "NMF15-TFIDF-2026-10-15-bereinigt"
TOPIC_DESCRIPTIONS = [
    (1, 'Straßenbeleuchtung', True, 'Ausfälle und Schäden an der Straßenbeleuchtung.', 'Mehrere Straßenlaternen sind ausgefallen, der Weg ist nachts dunkel.'),
    (2, 'Abgestellte Fahrräder', True, 'Abgestellte, herrenlose oder beschädigte Fahrräder.', 'Ein beschädigtes Fahrrad ist seit mehreren Wochen unverändert abgestellt.'),
    (3, 'Müllablagerungen', True, 'Abgelagerter Abfall, unter anderem an Containern und auf öffentlichen Flächen.', 'Neben den Glascontainern wurden Abfälle abgelegt.'),
    (4, 'Gehwege', True, 'Meldungen zu Gehwegen, etwa Bewuchs, abgestellte Gegenstände oder Einschränkungen der Nutzung.', 'Eine Hecke ragt in den Gehweg und verengt den Durchgang.'),
    (5, 'Straßen, Beleuchtung und Fahrzeuge', False, 'Unterschiedliche Anliegen zu Straßen, Beleuchtung, Bewuchs und Fahrzeugen ohne einheitlichen Schwerpunkt.', ''),
    (6, 'Beschilderung', True, 'Beschädigte, unleserliche oder fehlende Schilder und Hinweise zur Beschilderung.', 'Ein Straßenschild ist nicht mehr lesbar.'),
    (7, 'Radwege', True, 'Meldungen zu Radwegen, insbesondere Bewuchs, Scherben und andere Hindernisse.', 'Bewuchs verengt den Radweg und erschwert das Passieren.'),
    (8, 'Ampeln und Straßenquerungen', True, 'Ampeln, Signalphasen und Anliegen beim Überqueren von Straßen.', 'Eine Fußgängerampel schaltet nicht auf Grün.'),
    (9, 'Bäume, Fahrräder und Gegenstände', False, 'Bäume und Äste, aber auch Fahrräder, Bänke und abgelegte Gegenstände ohne einheitlichen Schwerpunkt.', ''),
    (10, 'Abgestellte Roller', True, 'Länger abgestellte oder beschädigte Roller, häufig ohne gültiges Versicherungskennzeichen.', 'Ein Roller ohne gültiges Versicherungskennzeichen steht längere Zeit am selben Ort.'),
    (11, 'Glasscherben', True, 'Glasscherben auf Wegen, an Haltestellen und auf anderen öffentlichen Flächen.', 'Auf einer öffentlichen Fläche liegen Glasscherben.'),
    (12, 'Sperrmüll', True, 'Abgestellter Sperrmüll, Möbel und größere Haushaltsgegenstände.', 'Auf dem Gehweg wurde Sperrmüll abgestellt.'),
    (13, 'Wasser, Abläufe und weitere Anliegen', False, 'Wasser und Abläufe, aber auch Verschmutzungen und Schäden ohne hinreichenden gemeinsamen Schwerpunkt.', ''),
    (14, 'Spielplätze, Mülleimer und Straßenschäden', False, 'Verschiedene Anliegen zu Spielplätzen, Müllbehältern, Wegen und Straßenschäden.', ''),
    (15, 'Fahrzeuge und Parkplätze', True, 'Abgestellte Fahrzeuge, fehlende Kennzeichen und weitere Anliegen auf Parkplätzen.', 'Auf einem Parkplatz steht seit längerer Zeit ein abgemeldetes Fahrzeug.'),
]
EXPECTED_TOPIC_WORDS = {
    1: ('beleuchtung', 'leuchten', 'kurve', 'licht', 'weitere', 'flackert', 'brachsengang', 'nachts', 'schief', 'hausnr'),
    2: ('fahrrad', 'herrenloses', 'moeglichkeit', 'sattel', 'drei', 'defektes', 'bewegt', 'fahrradstaendern', 'vorderrad', 'bushaltestelle'),
    3: ('müll', 'illegaler', 'glascontainer', 'altglascontainer', 'wald', 'illegale', 'abgeladen', 'mülleimer', 'illegal', 'altglascontainern'),
    4: ('gehweg', 'wächst', 'hecke', 'zugewachsen', 'ragt', 'fußgänger', 'rad', 'zugewuchert', 'briel', 'laub'),
    5: ('straße', 'entlang', 'leuchten', 'licht', 'abgemeldeter', 'scherben', 'auto', 'steig', 'gottlieber', 'erhebliche'),
    6: ('schild', 'boden', 'umgefahren', 'erkennen', 'verdreht', 'angefahren', 'unkentlich', 'unkenntlich', 'umgestürzt', 'ausgeblichen'),
    7: ('radweg', 'scherben', 'zugewachsen', 'entlang', 'fußgänger', 'radfahrer', 'stadtauswaerts', 'uebergang', 'verteilt', 'ausweichen'),
    8: ('ampel', 'fußgänger', 'grün', 'kreuzung', 'rot', 'licht', 'druecker', 'radfahrer', 'schaltet', 'autos'),
    9: ('baum', 'rad', 'umgestürzt', 'bank', 'matratze', 'quer', 'umgefallen', 'versperrt', 'äste', 'umgestuerzter'),
    10: ('roller', 'gueltiges', 'versicherungsschild', 'stadelhofgasse', 'bruder', 'klaus', 'teilweise', 'fahrradstaendern', 'anfang', 'ausgeschlachteter'),
    11: ('glasscherben', 'rad', 'radweg', 'kurve', 'platz', 'bushaltestelle', 'meter', 'müllsack', 'brunnen', 'muelltueten'),
    12: ('sperrmüll', 'illegaler', 'illegal', 'altglas', 'verschenken', 'alten', 'waschmaschine', 'matratze', 'schwarze', 'altglascontainern'),
    13: ('wasser', 'verstopft', 'läuft', 'ab', 'ablauf', 'gulli', 'regen', 'bushaltestelle', 'dadurch', 'scooter'),
    14: ('spielplatz', 'mülleimer', 'schlagloch', 'überfüllt', 'offen', 'wasserpumpe', 'tiefe', 'bushaltestelle', 'abgeladen', 'verteilt'),
    15: ('fahrzeug', 'parkplatz', 'kennzeichen', 'strassenrand', 'abgemeldetes', 'oeffentlichen', 'auto', 'reifen', 'zulassung', 'abgemeldet'),
}


def describe_selected_topics(summary, definitions, expected_words):
    """Namen und Beschreibungen mit der unveränderten Themenübersicht verbinden."""
    required = {"Topic", "Top-Begriffe", "Anzahl Meldungen", "Anteil Prozent", "Bezugsmenge Meldungen"}
    if not required.issubset(summary.columns) or not summary.columns.is_unique:
        raise ValueError("Die technische Themenübersicht ist unvollständig.")
    if summary["Topic"].isna().any() or summary["Topic"].duplicated().any():
        raise ValueError("Jedes Topic muss genau einmal in der Themenübersicht stehen.")
    labels = pd.DataFrame(definitions, columns=[
        "Topic", "Thema", "In Ergebnisübersicht", "Kurzbeschreibung", "Beispiel (sinngemäß)"
    ])
    if labels["Topic"].isna().any() or labels["Topic"].duplicated().any():
        raise ValueError("Jedes Topic benötigt genau eine Beschriftung.")
    if set(summary["Topic"]) != set(expected_words) or set(labels["Topic"]) != set(expected_words):
        raise ValueError("Die Beschriftungen passen nicht zu den vorhandenen Topics.")
    if labels["Thema"].isna().any() or labels["Thema"].str.strip().eq("").any():
        raise ValueError("Mindestens eine Themenbezeichnung fehlt.")
    if not labels["In Ergebnisübersicht"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Die Auswahl für die Ergebnisübersicht muss aus Wahrheitswerten bestehen.")
    for _, row in summary.iterrows():
        actual = tuple(word.strip() for word in str(row["Top-Begriffe"]).split(","))
        if actual != expected_words[int(row["Topic"])]:
            raise ValueError("Die Topic-Begriffe haben sich geändert. Beschriftungen erneut prüfen.")
    result = summary.merge(labels, on="Topic", how="left", validate="one_to_one")
    result["Einordnung"] = np.where(
        result["In Ergebnisübersicht"], "Schwerpunkt erkennbar", "Thematisch uneinheitlich"
    )
    return result.sort_values("Topic", kind="stable").reset_index(drop=True)

def build_result_view(labeled_topics, assignments, denominator):
    """Alle Modellgruppen erhalten, benennbare Schwerpunkte separat anzeigen."""
    if isinstance(denominator, bool) or not isinstance(denominator, (int, np.integer)) or denominator < 1:
        raise ValueError("Die Bezugsgröße muss eine positive ganze Zahl sein.")
    if len(assignments) != denominator or not assignments.index.is_unique or not assignments.columns.is_unique:
        raise ValueError("Meldungen und Bezugsgröße passen nicht zusammen.")
    required = {"Topic", "Status", "Dominanz", "Gleicher Höchstwert"}
    if not required.issubset(assignments.columns):
        raise ValueError("Die Zuordnungen benötigen Topic, Status und Dominanz.")
    if assignments["Status"].isna().any():
        raise ValueError("Für mindestens eine Meldung fehlt der Zuordnungsstatus.")
    known = set(labeled_topics["Topic"])
    if not set(assignments["Topic"].dropna()).issubset(known):
        raise ValueError("Eine Zuordnung verweist auf ein unbekanntes Topic.")
    assigned = assignments["Topic"].notna()
    if not np.array_equal(assigned.to_numpy(), assignments["Status"].eq("Topic zugeordnet").to_numpy()):
        raise ValueError("Topic und Zuordnungsstatus widersprechen sich.")
    values = assignments.loc[assigned, "Dominanz"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or (values > 1 + 1e-12).any():
        raise ValueError("Ungültige Dominanzwerte in den zugeordneten Meldungen.")
    if assignments.loc[~assigned, "Dominanz"].notna().any():
        raise ValueError("Eine Meldung ohne Topic darf keinen Dominanzwert besitzen.")
    full = labeled_topics.copy(deep=True)
    counts = assignments["Topic"].value_counts().reindex(full["Topic"], fill_value=0)
    if not np.array_equal(counts.to_numpy(), full["Anzahl Meldungen"].to_numpy()):
        raise ValueError("Die Häufigkeiten passen nicht zu den tatsächlichen Zuordnungen.")
    if not full["Bezugsmenge Meldungen"].eq(denominator).all() or not np.allclose(
        full["Anteil Prozent"], 100 * full["Anzahl Meldungen"] / denominator, rtol=0, atol=1e-10
    ):
        raise ValueError("Die Prozentanteile verwenden eine abweichende Bezugsgröße.")
    # Zusatzdiagnose in der technischen Tabelle, keine Filterung der Ergebnisübersicht.
    high = assignments.loc[assigned & assignments["Dominanz"].gt(0.5), "Topic"].value_counts()
    full["Davon Dominanz größer 0,5"] = full["Topic"].map(high).fillna(0).astype(int)
    full["Davon Dominanz höchstens 0,5"] = full["Anzahl Meldungen"] - full["Davon Dominanz größer 0,5"]
    shown = full.loc[full["In Ergebnisübersicht"]].sort_values(
        ["Anzahl Meldungen", "Topic"], ascending=[False, True], kind="stable"
    )
    table = shown[["Thema", "Anzahl Meldungen", "Anteil Prozent", "Kurzbeschreibung", "Beispiel (sinngemäß)"]].copy()
    table = table.reset_index(drop=True)
    table.insert(0, "Rang", np.arange(1, len(table) + 1))
    n_shown = int(table["Anzahl Meldungen"].sum())
    n_mixed = int(full.loc[~full["In Ergebnisübersicht"], "Anzahl Meldungen"].sum())
    n_unassigned = int((~assigned).sum())
    if n_shown + n_mixed + n_unassigned != denominator:
        raise RuntimeError("Die Aufteilung umfasst nicht genau alle Meldungen nach Preprocessing.")
    coverage = pd.DataFrame({
        "Bereich": ["In der Themenübersicht dargestellt", "Nicht nach Themen aufgeschlüsselt"],
        "Anzahl Meldungen": [n_shown, n_mixed + n_unassigned],
    })
    coverage["Anteil Prozent"] = 100 * coverage["Anzahl Meldungen"] / denominator
    coverage["Bezugsmenge Meldungen"] = denominator
    reasons = pd.DataFrame({
        "Grund": ["Kein einheitlicher Schwerpunkt des Topics", "Keine Topic-Zuordnung"],
        "Anzahl Meldungen": [n_mixed, n_unassigned],
    })
    reasons["Anteil Prozent"] = 100 * reasons["Anzahl Meldungen"] / denominator
    return full, table, coverage, reasons

def german_number(value, decimals=0):
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")

def plot_result_view(table, coverage, output_path, period_label):
    """Themenhäufigkeiten zeigen und den nicht aufgeschlüsselten Anteil separat ausweisen."""
    if table.empty or table["Thema"].duplicated().any():
        raise ValueError("Für die Grafik werden eindeutige Themen mit Meldungen benötigt.")
    fig, ax = plt.subplots(figsize=(13.8, 10.4))
    positions = np.arange(len(table))
    bars = ax.barh(positions, table["Anzahl Meldungen"].to_numpy(), height=0.65)
    ax.set_yticks(positions, labels=table["Thema"].tolist(), fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(0, float(table["Anzahl Meldungen"].max()) * 1.27)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("Anzahl der automatisch zugeordneten Meldungen", fontsize=11.5, labelpad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    for bar, count, percentage in zip(bars, table["Anzahl Meldungen"], table["Anteil Prozent"]):
        ax.text(bar.get_width() + table["Anzahl Meldungen"].max() * 0.017,
                bar.get_y() + bar.get_height() / 2,
                f"{german_number(count)}  ({german_number(percentage, 1)} %)",
                va="center", fontsize=11)
    fig.suptitle("Gemeldete Mängel nach Themenbereich", x=0.04, y=0.965,
                 ha="left", fontsize=22, fontweight="bold")
    fig.text(0.04, 0.908,
             f"Mängelmelder Konstanz · {period_label} · Automatische Themenanalyse",
             ha="left", fontsize=12)

    # Der graue Bereich gehört nicht zur Rangfolge der Themen.
    fig.add_artist(Rectangle((0.04, 0.076), 0.93, 0.111,
                             transform=fig.transFigure, facecolor="0.94", edgecolor="none"))
    shown, other = coverage.iloc[0], coverage.iloc[1]
    for row, label, y in (
        (shown, "Nach Themen aufgeschlüsselt", 0.150),
        (other, "Nicht nach Themen aufgeschlüsselt", 0.111),
    ):
        fig.text(0.058, y, label, ha="left", va="center", fontsize=12, fontweight="bold")
        fig.text(0.62, y,
                 f"{german_number(row['Anzahl Meldungen'])} Meldungen ({german_number(row['Anteil Prozent'], 1)} %)",
                 ha="left", va="center", fontsize=12, fontweight="bold")
    fig.text(0.04, 0.035,
             "Die Reihenfolge zeigt die Häufigkeit der Meldungen, nicht ihre Dringlichkeit.",
             ha="left", fontsize=11)
    fig.subplots_adjust(left=0.285, right=0.97, top=0.857, bottom=0.26)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig

def result_period(date_values: pd.Series):
    """Deutsche Monatsangaben aus den Erstellungsdaten der analysierten Meldungen bilden."""
    report_dates = pd.to_datetime(date_values, format="%Y-%m-%d", errors="raise")
    if report_dates.empty or report_dates.isna().any():
        raise ValueError("Für den Zeitraum werden gültige Erstellungsdaten der Meldungen benötigt.")
    report_start, report_end = report_dates.min(), report_dates.max()
    month_names = ("", "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
                   "August", "September", "Oktober", "November", "Dezember")
    period_label = (f"{month_names[report_start.month]} {report_start.year} bis "
                    f"{month_names[report_end.month]} {report_end.year}")

    return report_start, report_end, period_label
