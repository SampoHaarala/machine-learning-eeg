import numpy as np


def make_split(metadata, cfg, seed):
    rng = np.random.default_rng(seed)
    subjects = np.array([m['subject'] for m in metadata])
    names = ('train', 'validation', 'test')
    splits = {k: [] for k in names}
    if cfg['mode'] == 'subject':
        available = sorted(set(subjects.tolist()))
        explicit = cfg['subjects']
        if explicit:
            groups = [list(map(str, explicit[k])) for k in names]
            flat = sum(groups, [])
            if len(set(flat)) != len(flat) or set(flat) != set(available):
                raise ValueError('Explicit split must be disjoint and cover every selected subject exactly once')
        else:
            counts = cfg['counts']
            if any(n < 1 for n in counts) or sum(counts) != len(available):
                raise ValueError(f'Split counts {counts} must sum to {len(available)} available subjects')
            shuffled = rng.permutation(available)
            groups = np.split(shuffled, np.cumsum(counts)[:-1])
        for key, group in zip(names, groups):
            splits[key] = np.flatnonzero(np.isin(subjects, group)).tolist()
    else:
        fractions = cfg['within_fractions']
        if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
            raise ValueError('Three positive within-subject fractions must sum to one')
        field = cfg['within_group']
        if field not in ('block', 'trial_id'):
            raise ValueError('Within-subject group must be block or trial_id')
        for subject in sorted(set(subjects)):
            ids = np.flatnonzero(subjects == subject)
            groups = sorted({metadata[i][field] for i in ids})
            if len(groups) < 3:
                raise ValueError(f'Subject {subject} needs at least 3 independent groups')
            groups = rng.permutation(groups)
            ntrain = max(1, min(len(groups)-2, int(len(groups)*fractions[0])))
            nval = max(1, min(len(groups)-ntrain-1, int(len(groups)*fractions[1])))
            partitions = np.split(groups, [ntrain, ntrain+nval])
            for key, group in zip(names, partitions):
                splits[key].extend(int(i) for i in ids if metadata[i][field] in group)
    if any(len(v) < 2 for v in splits.values()):
        raise ValueError('Every split needs at least two samples')
    sets = [set(v) for v in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise AssertionError('Trial leakage')
    if cfg['mode'] == 'subject':
        ss = [set(subjects[v]) for v in splits.values()]
        if any(ss[i] & ss[j] for i in range(3) for j in range(i)):
            raise AssertionError('Subject leakage')
    report = {k: {'indices': v, 'subjects': sorted(set(subjects[v].tolist())),
                  'trials': [f'{subjects[i]}/{metadata[i]["trial_id"]}' for i in v]}
              for k, v in splits.items()}
    return {k: np.array(v) for k, v in splits.items()}, report
