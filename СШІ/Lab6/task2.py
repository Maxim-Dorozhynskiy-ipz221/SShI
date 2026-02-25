import argparse
from collections import Counter, defaultdict

DATASET = [
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

ATTRS = ["outlook", "humidity", "wind"]

QUERY = {"outlook": "Overcast", "humidity": "High", "wind": "Weak"}

def collect_counts(rows):
    cls_counts = Counter(r["play"] for r in rows)
    feat_stats = {f: {c: Counter() for c in cls_counts} for f in ATTRS}
    domains = {f: set() for f in ATTRS}
    for r in rows:
        c = r["play"]
        for f in ATTRS:
            v = r[f]
            feat_stats[f][c][v] += 1
            domains[f].add(v)
    domains = {f: sorted(list(vs)) for f, vs in domains.items()}
    return cls_counts, feat_stats, domains

def prob_tables(cls_counts, feat_stats, domains, alpha=1.0):
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

def posterior_probs(x, cls_counts, like):
    N = sum(cls_counts.values())
    scores = {}
    for c, cnt in cls_counts.items():
        prior = cnt / N
        cond = 1.0
        for f, v in x.items():
            cond *= like[f][c].get(v, 0.0)
        scores[c] = prior * cond
    s = sum(scores.values())
    if s > 0:
        for c in scores:
            scores[c] /= s
    return scores

def main():
    ap = argparse.ArgumentParser(description="Naive Bayes для прикладу (Rain, High, Strong)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()

    cls_counts, feat_stats, domains = collect_counts(DATASET)
    like = prob_tables(cls_counts, feat_stats, domains, alpha=args.alpha)

    post = posterior_probs(QUERY, cls_counts, like)
    pred = max(post, key=post.get)

    if not args.no_header:
        print("Приклад: Outlook=Rain, Humidity=High, Wind=Strong")
    print(f"alpha={args.alpha}")
    print(f"P(Yes|x) = {post.get('Yes',0):.6f}")
    print(f"P(No |x) = {post.get('No',0):.6f}")
    print(f"Прогноз : {pred}")

if __name__ == "__main__":
    main()