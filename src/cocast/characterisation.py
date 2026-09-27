import json
import numpy as np
from . import schema


def characterise(table, run, manifest):
    """Achieved severity distributions and agreement with assigned tiers."""
    scores = schema.score_table(table)
    rows = []
    for domain in schema.DISORDER_NAMES:
        tiers = scores[schema.tier_col(domain)]
        for tier in schema.TIER_LABELS:
            rows.append(dict(family='characterisation', domain=domain,
                             metric=f'{tier}_prevalence_percent', value=100 * float((tiers == tier).mean())))
    if manifest.get('kind') == 'baseline':
        return rows
    accepted = {}
    for line in (run / 'responses.jsonl').open():
        entry = json.loads(line)
        validation = entry.get('validation', {})
        if validation.get('valid') and 'tier_match' in validation:
            key = entry['request_id']
            if key in accepted:
                raise ValueError('Duplicate successful questionnaire response in journal.')
            accepted[key] = validation['tier_match']
    if len(accepted) != len(table) * len(schema.DISORDER_NAMES):
        raise ValueError('Assigned-tier agreement requires all questionnaire responses.')
    for domain in [*schema.DISORDER_NAMES, 'all_domains']:
        matches = [v for k, v in accepted.items() if domain == 'all_domains' or k.rsplit(':', 1)[1] == domain]
        rows.append(dict(family='characterisation', domain=domain, metric='tier_agreement_percent',
                         value=100 * float(np.mean(matches))))
    return rows
