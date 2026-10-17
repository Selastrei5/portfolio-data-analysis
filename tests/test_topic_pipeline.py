"""Funktionstests für Datenaufbereitung, Zuordnung und Ergebnisübersicht.

Aufruf aus dem Projektordner: python -m unittest discover -s tests -v
Die Tests verwenden künstliche Daten und benötigen die Original-CSV nicht.
"""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src import topic_pipeline as tp, wordlists, result_view as rv


class TextProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lookup = tp.build_normalization_lookup(
            wordlists.CANONICAL_TERMS_FINAL, wordlists.NORMALIZATION_GROUPS
        )
        cls.stopwords = {tp.normalization_key(w) for w in wordlists.STOPWORDS_FINAL}
        cls.stopwords -= {tp.normalization_key(w) for w in wordlists.NORMALIZATION_KEEP}

    def clean(self, text):
        return tp.clean_text(text, lookup=self.lookup, stopwords=self.stopwords,
                             min_token_length=2, phrases=wordlists.NORMALIZATION_PHRASES)

    def test_existing_text_examples(self):
        examples = {
            "Lampe defekt": "beleuchtung", "Müll": "müll", "Bitte danke": "",
            "Leuchte defekt": "beleuchtung",
            "Die Leuchten leuchten nicht": "leuchten leuchten",
            "Die Ampeln leuchten nicht": "ampel leuchten",
            "Müllentsorgung Vermuellung": "müll müll",
            "Fahrrädern Fahrraedern": "fahrrad fahrrad",
            "Straßenleuchte Strassenleuchte": "beleuchtung beleuchtung",
            "Sehr geehrte Damen und Herren, Müll": "müll",
            "Sofas Glasflaschen": "sofa glasflasche",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(self.clean(text), expected)

    def test_lookup_resolves_chains(self):
        lookup = tp.build_normalization_lookup({"container": "müllcontainer", "müllcontainer": "müll"}, {})
        self.assertEqual(lookup["container"], "müll")

    def test_lookup_rejects_cycles(self):
        with self.assertRaises(ValueError):
            tp.build_normalization_lookup({"eins": "zwei", "zwei": "eins"}, {})

    def test_lookup_rejects_conflicting_spellings(self):
        with self.assertRaises(ValueError):
            tp.build_normalization_lookup({"müll": "abfall", "muell": "sofa"}, {})

    def test_existing_location_exceptions(self):
        for word in ("straße", "gehweg", "radweg", "fahrradweg", "spielplatz", "parkplatz"):
            with self.subTest(word=word):
                self.assertFalse(tp.is_location_token(word))
        self.assertTrue(tp.is_location_token("hauptstrasse"))

    def test_url_markup_removal(self):
        self.assertEqual(self.clean('<b>Müll</b> https://example.invalid/baum'), 'müll')


class RecordTests(unittest.TestCase):
    def test_missing_empty_and_duplicate_rows(self):
        source = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 8],
            "text": [None, np.nan, pd.NA, "", "  ", " Müll ", "Müll", "Lampe defekt", "Lampe defekt"],
        })
        before = source.copy(deep=True)
        result, counts = tp.prepare_records(source, "text", 1, "id")
        self.assertEqual(result.id.tolist(), [6, 7, 8])
        self.assertEqual(result.text.tolist(), ["Müll", "Müll", "Lampe defekt"])
        self.assertEqual(counts, {"input": 9, "duplicates": 1, "missing": 3, "empty": 2,
                                  "raw_short": 0, "remaining": 3, "text_repetitions": 1})
        pd.testing.assert_frame_equal(source, before)

    def test_conflicting_ids(self):
        with self.assertRaises(ValueError):
            tp.prepare_records(pd.DataFrame({"id": [1, 1], "text": ["Müll", "Lampe"]}), "text", 1, "id")

    def test_missing_ids(self):
        for bad in (None, "", "  "):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tp.prepare_records(pd.DataFrame({"id": [bad], "text": ["Müll"]}), "text", 1, "id")

    def test_missing_columns(self):
        with self.assertRaises(ValueError):
            tp.prepare_records(pd.DataFrame({"id": [1]}), "text", 1, "id")

    def test_minimum_length_validation(self):
        for minimum in (True, 0, -1, 1.5):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                tp.prepare_records(pd.DataFrame({"id": [1], "text": ["Müll"]}), "text", minimum, "id")


class MatrixAndAssignmentTests(unittest.TestCase):
    def test_selection_preserves_index_and_inputs(self):
        docs = pd.DataFrame({"id": [1, 2, 3]}, index=[10, 20, 30])
        matrix = csr_matrix([[1., 0.], [0., 0.], [0., 2.]])
        original = matrix.copy()
        mask, fit, count, tfidf, without = tp.select_model_documents(docs, matrix, matrix)
        np.testing.assert_array_equal(mask, [True, False, True])
        self.assertEqual(fit.index.tolist(), [10, 30])
        self.assertEqual(without.index.tolist(), [20])
        self.assertEqual(count.shape, (2, 2))
        self.assertEqual((matrix != original).nnz, 0)

    def test_different_zero_rows_rejected(self):
        with self.assertRaises(ValueError):
            tp.select_model_documents(pd.DataFrame(index=[1, 2]), csr_matrix([[1.], [0.]]), csr_matrix([[1.], [2.]]))

    def test_invalid_feature_values(self):
        for val in (-1., float('nan')):
            with self.subTest(val=val), self.assertRaises(ValueError):
                tp.select_model_documents(pd.DataFrame(index=[1]), csr_matrix([[val]]), csr_matrix([[val]]))

    def test_scaling_preserves_reconstruction_and_inputs(self):
        w = np.array([[1., 2.], [0., 0.]])
        h = np.array([[10., 0.], [0., 1.]])
        before = w.copy(), h.copy()
        scaled, unit = tp.normalize_nmf_for_assignment(w, h)
        np.testing.assert_allclose(scaled @ unit, w @ h)
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.)
        np.testing.assert_array_equal(w, before[0])
        np.testing.assert_array_equal(h, before[1])
        factors = np.array([.01, 100.])
        other, _ = tp.normalize_nmf_for_assignment(w / factors, h * factors[:, None])
        np.testing.assert_allclose(scaled, other)
        np.testing.assert_array_equal(scaled[1], [0., 0.])

    def test_zero_component_rejected(self):
        with self.assertRaises(ValueError):
            tp.normalize_nmf_for_assignment(np.ones((1, 2)), np.zeros((2, 3)))

    def test_negative_weights_rejected(self):
        with self.assertRaises(ValueError):
            tp.assign_topics(np.array([[-1., .5]]), np.array([True]), pd.Index([1]))

    def test_assignment_and_missing_cases(self):
        result = tp.assign_topics(np.array([[.2, .8], [0., 0.]]), np.array([True, False, True]), pd.Index([10, 20, 30]))
        self.assertEqual(result.loc[10, "Topic"], 2)
        self.assertTrue(result.loc[[20, 30], "Topic"].isna().all())
        self.assertEqual(result.loc[20, "Status"], "Keine Merkmale im Vokabular")
        self.assertEqual(result.loc[30, "Status"], "Keine positiven Topic-Gewichte")

    def test_tie_and_zero_dominance(self):
        weights = np.array([[.5, .5], [0., 0.], [1e308, 1e308]])
        dominance, ties = tp.topic_dominance(weights)
        np.testing.assert_allclose(dominance[[0, 2]], [.5, .5])
        self.assertTrue(np.isnan(dominance[1]))
        np.testing.assert_array_equal(ties, [True, False, True])
        result = tp.assign_topics(weights, np.array([True, True, True]), pd.Index([1, 2, 3]))
        self.assertEqual(result.loc[1, 'Topic'], 1)
        self.assertTrue(result.loc[1, 'Gleicher Höchstwert'])

    def test_empty_training_selection(self):
        result = tp.assign_topics(np.empty((0, 2)), np.array([False, False]), pd.Index([1, 2]))
        self.assertTrue(result.Topic.isna().all())


class CategoryTests(unittest.TestCase):
    def test_permuted_perfect_groups(self):
        scores = tp.comparison_category_scores(pd.Series([2, 2, 1, 1]), pd.Series(['a', 'a', 'b', 'b']))
        for key in ('NMI', 'AMI', 'ARI', 'Reinheit'):
            self.assertAlmostEqual(scores[key], 1.)

    def test_missing_categories_reported(self):
        scores = tp.comparison_category_scores(pd.Series([1, 2, 2]), pd.Series(['a', None, ' ']))
        self.assertEqual(scores['Bewertete Meldungen'], 1)
        self.assertEqual(scores['Ohne Kategorie'], 2)
        self.assertTrue(np.isnan(scores['AMI']))

    def test_all_missing_categories(self):
        scores = tp.comparison_category_scores(pd.Series([1, 2]), pd.Series([None, None]))
        self.assertEqual(scores['Bewertete Meldungen'], 0)
        self.assertTrue(np.isnan(scores['Reinheit']))

    def test_misaligned_categories_rejected(self):
        with self.assertRaises(ValueError):
            tp.comparison_category_scores(pd.Series([1], index=[1]), pd.Series(['a'], index=[2]))


class ResultViewTests(unittest.TestCase):
    def setUp(self):
        self.summary = pd.DataFrame({'Topic': [1, 2], 'Top-Begriffe': ['müll, abfall', 'straße, licht'],
            'Anzahl Meldungen': [2, 1], 'Anteil Prozent': [50., 25.], 'Bezugsmenge Meldungen': [4, 4]})
        self.definitions = [(1, 'Müll', True, 'Abfall', 'Abfall liegt dort.'), (2, 'Andere', False, 'Verschiedenes', '')]
        self.expected = {1: ('müll', 'abfall'), 2: ('straße', 'licht')}
        self.assignments = tp.assign_topics(np.array([[.9, .1], [.5, .5], [.1, .9]]), np.array([True, True, True, False]), pd.Index(range(4)))
        self.assignments['Dominanz'] = [.9, .5, .9, np.nan]
        self.labels = rv.describe_selected_topics(self.summary, self.definitions, self.expected)

    def test_coverage_no_dominance_filter_and_no_mutation(self):
        original = self.assignments.copy(deep=True)
        full, shown, coverage, reasons = rv.build_result_view(self.labels, self.assignments, 4)
        self.assertEqual(shown['Anzahl Meldungen'].tolist(), [2])
        self.assertEqual(coverage['Anzahl Meldungen'].tolist(), [2, 2])
        self.assertEqual(reasons['Anzahl Meldungen'].tolist(), [1, 1])
        self.assertEqual(full.loc[0, 'Davon Dominanz größer 0,5'], 1)
        self.assertEqual(full.loc[0, 'Davon Dominanz höchstens 0,5'], 1)
        pd.testing.assert_frame_equal(original, self.assignments)

    def test_repeated_result_view_is_idempotent(self):
        first = rv.build_result_view(self.labels, self.assignments, 4)
        second = rv.build_result_view(first[0], self.assignments, 4)
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(a, b)

    def test_changed_topic_words_rejected(self):
        altered = self.summary.copy()
        altered.loc[0, 'Top-Begriffe'] = 'fahrrad, licht'
        with self.assertRaises(ValueError):
            rv.describe_selected_topics(altered, self.definitions, self.expected)

    def test_duplicate_labels_rejected(self):
        with self.assertRaises(ValueError):
            rv.describe_selected_topics(self.summary, self.definitions + [self.definitions[0]], self.expected)

    def test_missing_label_rejected(self):
        with self.assertRaises(ValueError):
            rv.describe_selected_topics(self.summary, self.definitions[:1], self.expected)

    def test_wrong_denominator_rejected(self):
        with self.assertRaises(ValueError):
            rv.build_result_view(self.labels, self.assignments, 3)

    def test_wrong_frequencies_rejected(self):
        altered = self.labels.copy()
        altered.loc[0, 'Anzahl Meldungen'] = 3
        with self.assertRaises(ValueError):
            rv.build_result_view(altered, self.assignments, 4)

    def test_status_mismatch_rejected(self):
        self.assignments.loc[0, 'Status'] = 'Keine Merkmale im Vokabular'
        with self.assertRaises(ValueError):
            rv.build_result_view(self.labels, self.assignments, 4)

    def test_period_in_german(self):
        start, end, label = rv.result_period(pd.Series(['2026-06-02', '2022-07-26']))
        self.assertEqual(label, 'Juli 2022 bis Juni 2026')
        self.assertEqual(start.date().isoformat(), '2022-07-26')
        self.assertEqual(end.date().isoformat(), '2026-06-02')

    def test_invalid_date_rejected(self):
        with self.assertRaises(ValueError):
            rv.result_period(pd.Series(['not a date']))

    def test_german_number_format(self):
        self.assertEqual(rv.german_number(1994), '1.994')
        self.assertEqual(rv.german_number(72.3, 1), '72,3')


if __name__ == '__main__':
    unittest.main()
