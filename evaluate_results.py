import json

from itertools import islice
from pathlib import Path


class Evaluator:
    """Evaluate the performance of a recommender system.

    Args:
        dataset: name of the dataset to use for evaluation.
    """

    def __init__(self, dataset):
        dataset_path = Path(__file__).parent / "data" / f"{dataset}.jsonl"
        self.records = {}
        self.references = {}
        self.scores = {}
        self.data = self._load_data(dataset_path)

    def _load_data(self, path):
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                recid = record["control_number"]
                self.records[recid] = record
                self.references[recid] = {
                    ref["record"]["$ref"].split("/")[-1]
                    for ref in record.get("references", [])
                    if "record" in ref
                }

    def evaluate(self, recommender_function, limit=10):
        """Run the evaluation.

        Args:
            recommender_function (Callable): function taking as input the
                record metadata, and returning an iterable of results (ordered by
                relevance).
            limit (int): number of results from the recommender_function to
                keep for evaluation.

        Returns:
            int: the score of the recommender on the dataset. A score of 1
                means that all recommendations (within the limit) were among
                the references.
        """

        for recid, record in self.records.items():
            predictions = set(islice(recommender_function(record), limit))
            try:
                self.scores[recid] = len(predictions & self.references[recid]) / min(
                    len(self.references[recid]), limit
                )
            except ZeroDivisionError:
                # No references
                pass

        return sum(self.scores.values()) / len(self.scores)


if __name__ == "__main__":

    def dummy_recommender(record):
        """Recommend the record itself and the first reference if linked."""
        results = [record["control_number"]]
        try:
            ref = record["references"][0]["record"]["$ref"].split("/")[-1]
            results.append(ref)
        except (IndexError, KeyError):
            pass
        return results

    evaluator = Evaluator("random-core")
    print("Score of the dummy recommender:", evaluator.evaluate(dummy_recommender))
