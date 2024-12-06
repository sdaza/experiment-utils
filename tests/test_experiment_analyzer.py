# test_experiment_analyzer.py
import unittest
from experiment_utils.experiment_analyzer import ExperimentAnalyzer

class TestExperimentAnalyzer(unittest.TestCase):
    def test_init(self):
        # Create a sample DataFrame
        data = [(1, 'a', 10), (2, 'b', 20)]
        df = spark.createDataFrame(data, ['id', 'variable', 'value'])

        # Test ExperimentAnalyzer initialization
        analyzer = ExperimentAnalyzer(df, ['variable'], 'id')
        self.assertEqual(analyzer.data.count(), 2)
        self.assertEqual(analyzer.outcomes, ['variable'])

    def test_get_effects(self):
        # Create a sample DataFrame
        data = [(1, 'a', 10), (2, 'b', 20)]
        df = spark.createDataFrame(data, ['id', 'variable', 'value'])

        # Test get_effects function
        analyzer = ExperimentAnalyzer(df, ['variable'], 'id')
        effects = analyzer.get_effects()
        self.assertEqual(effects.count(), 1)

if __name__ == '__main__':
    unittest.main()