# Copyright (C) 2020 CERN.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Fetch data for the JSON benchmark."""
import json
import requests

from itertools import islice

from backoff import expo, on_exception
from tqdm import tqdm


INSPIRE_LITERATURE_API_ENDPOINT = "https://inspirehep.net/api/literature"

session = requests.Session()
session.headers[
    "user-agent"
] += "INSPIRE API Client (mailto:micha.moshe.moskovic@cern.ch)"


@on_exception(expo, requests.exceptions.HTTPError, max_tries=10)
def perform_inspire_literature_search(query=None, fields=(), **kwargs):
    """Perform the search query on INSPIRE.
    Args:
        query (str): the search query to get the results for.
        fields (iterable): a list of fields to return.
        kwargs (dict): extra arguments to pass to the query.
    Yields:
        dict: the json response for every record.
    """
    response = session.get(
        INSPIRE_LITERATURE_API_ENDPOINT,
        params={"q": query, "fields": ",".join(fields), **kwargs},
    )
    response.raise_for_status()
    content = response.json()

    for result in content["hits"]["hits"]:
        yield result["metadata"]

    while "next" in content.get("links", {}):
        response = session.get(content["links"]["next"])
        response.raise_for_status()
        content = response.json()

        for result in content["hits"]["hits"]:
            yield result["metadata"]


def searches():
    return {
        "random-core": perform_inspire_literature_search(
            "core:true and references.record.$ref:**", earliest_date="2015--2020"
        ),
    }


def dump_search(name, results, count=1000):
    with open(f"data/{name}.jsonl", "w") as f:
        for result in tqdm(islice(results, count), desc=name, total=count):
            json.dump(result, f)
            f.write("\n")


if __name__ == "__main__":
    for name, result in searches().items():
        dump_search(name, result)
