"""Fixed-query manifests contain IDs and order, never unrevealed labels."""
import csv
from pathlib import Path


def read_queries(path, data, future_months, beta, updater):
    import math
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'run_id', 'month', 'sample_id', 'query_order'}.issubset(reader.fieldnames or []):
            raise ValueError('Invalid query manifest columns')
        rows = list(reader)
    if len({r['run_id'] for r in rows}) > 1:
        raise ValueError('Query manifest must describe one run')
    if any(not r['run_id'].strip() for r in rows):
        raise ValueError('Query run_id must be nonempty')
    by_id = {str(v): i for i, v in enumerate(data.ids)}
    result = {m: [] for m in future_months}
    seen = set()
    for row in rows:
        month, sample_id = row['month'], row['sample_id']
        if month not in result or month == future_months[-1]:
            raise ValueError('Queries outside eligible months; final month is evaluation only')
        if sample_id not in by_id or data.months[by_id[sample_id]] != month or sample_id in seen:
            raise ValueError('Duplicate query or ID outside current month')
        seen.add(sample_id)
        result[month].append((int(row['query_order']), by_id[sample_id]))
    for month, entries in result.items():
        entries.sort()
        if [e[0] for e in entries] != list(range(len(entries))):
            raise ValueError('query_order must be contiguous from zero per month')
        budget = 0 if updater == 'frozen' else math.ceil(beta * sum(data.months == month))
        if len(entries) > budget:
            raise ValueError('Query manifest exceeds label budget')
        if updater == 'full_oracle':
            raise ValueError('full_oracle is incompatible with fixed-query replay')
        result[month] = [i for _, i in entries]
    return result
