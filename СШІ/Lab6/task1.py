import argparse
from collections import Counter, defaultdict
from pprint import pprint

def dataset_play_tennis():
    return [
        {"outlook": "Sunny", "humidity": "High", "wind": "Weak", "play": "No"},
        {"outlook": "Sunny", "humidity": "High", "wind": "Strong", "play": "No"},
        {"outlook": "Overcast", "humidity": "High", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "High", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "Normal", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "Normal", "wind": "Strong", "play": "No"},
        {"outlook": "Overcast", "humidity": "Normal", "wind": "Strong", "play": "Yes"},
        {"outlook": "Sunny", "humidity": "High", "wind": "Weak", "play": "No"},
        {"outlook": "Sunny", "humidity": "Normal", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "High", "wind": "Weak", "play": "Yes"},
        {"outlook": "Sunny", "humidity": "Normal", "wind": "Strong", "play": "Yes"},
        {"outlook": "Overcast", "humidity": "High", "wind": "Strong", "play": "Yes"},
        {"outlook": "Overcast", "humidity": "Normal", "wind": "Weak", "play": "Yes"},
        {"outlook": "Rain", "humidity": "High", "wind": "Strong", "play": "No"},
    ]

def frequency_tables(rows, features, target="play"):
    cls_counts = Counter(r[target] for r in rows)
    feat_stats = {f: {c: Counter() for c in cls_counts} for f in features}
    domains = {f: set() for f in features}
    for r in rows:
        c = r[target]
        for f in features:
            v = r[f]
            feat_stats[f][c][v] += 1
            domains[f].add(v)
    return cls_counts, feat_stats, {f: sorted(list(vs)) for f, vs in domains.items()}

def prob_tables(cls_counts, feat_stats, domains, alpha=0.0):
    like = defaultdict(lambda: defaultdict(dict))
    for f, per_class in feat_stats.items():
        K = len(domains[f])
        for c, counts in per_class.items():
            total_c = cls_counts[c]
            for v in domains[f]:
                num = counts.get(v, 0) + alpha
                den = total_c + alpha * K
                like[f][c][v] = num / den
    return like

def posterior_probs(x, cls_counts, like_table):
    N = sum(cls_counts.values())
    scores = {}
    for c, cnt in cls_counts.items():
        prior = cnt / N
        cond = 1.0
        for f, v in x.items():
            cond *= like_table[f][c].get(v, 0.0)
        scores[c] = prior * cond
    s = sum(scores.values())
    if s > 0:
        for c in scores:
            scores[c] /= s
    return scores

def main():
    parser = argparse.ArgumentParser(description="Naive Bayes для 'Play Tennis'")
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    rows = dataset_play_tennis()
    features = ["outlook", "humidity", "wind"]

    cls_counts, feat_stats, domains = frequency_tables(rows, features)
    like = prob_tables(cls_counts, feat_stats, domains, alpha=args.alpha)

    print("Кількість класів:", cls_counts)
    print("\nДомен ознак:")
    pprint(domains)

    print("\nЙмовірності (P(feature=value | class)):")
    pprint({f: {c: dict(d) for c, d in like[f].items()} for f in features})

    x = {"outlook": "Rain", "humidity": "High", "wind": "Weak"}
    post = posterior_probs(x, cls_counts, like)
    print("\nПриклад (Rain, High, Weak):", post)
    pred = max(post, key=post.get)
    print("Прогноз:", pred)

if __name__ == "__main__":
    main()