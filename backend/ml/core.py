"""Minimal, dependency-free ML primitives used by the prototype.

Pure Python keeps the demo portable (any laptop acting as the MEC node). The
interfaces mirror scikit-learn so they can be swapped for XGBoost / LightGBM /
PyTorch models when real operator data is available.
"""
import math
import re
from collections import Counter, defaultdict


# --------------------------------------------------------------------------
# Gradient boosting (histogram-based regression trees)
# --------------------------------------------------------------------------
class GBMRegressor:
    def __init__(self, n_estimators=80, learning_rate=0.15, max_depth=3, bins=24, min_leaf=20):
        self.n, self.lr, self.depth, self.bins, self.min_leaf = n_estimators, learning_rate, max_depth, bins, min_leaf

    def _bin_edges(self, X):
        edges = []
        for j in range(len(X[0])):
            col = sorted(r[j] for r in X)
            qs = sorted({col[int(len(col) * k / self.bins)] for k in range(1, self.bins)})
            edges.append(qs)
        return edges

    def _binize(self, row):
        out = []
        for j, v in enumerate(row):
            e = self.edges[j]
            lo, hi = 0, len(e)
            while lo < hi:
                m = (lo + hi) // 2
                if v < e[m]:
                    hi = m
                else:
                    lo = m + 1
            out.append(lo)
        return out

    def _grow(self, idx, B, g, depth):
        s = sum(g[i] for i in idx)
        leaf = {"v": s / len(idx)}
        if depth == 0 or len(idx) < 2 * self.min_leaf:
            return leaf
        best = None
        total_n = len(idx)
        for j in range(len(self.edges)):
            nb = len(self.edges[j]) + 1
            hs, hc = [0.0] * nb, [0] * nb
            for i in idx:
                b = B[i][j]
                hs[b] += g[i]
                hc[b] += 1
            ls, lc = 0.0, 0
            for b in range(nb - 1):
                ls += hs[b]
                lc += hc[b]
                rc = total_n - lc
                if lc < self.min_leaf or rc < self.min_leaf:
                    continue
                gain = ls * ls / lc + (s - ls) ** 2 / rc
                if best is None or gain > best[0]:
                    best = (gain, j, b)
        if best is None or best[0] <= s * s / total_n + 1e-12:
            return leaf
        _, j, b = best
        L = [i for i in idx if B[i][j] <= b]
        R = [i for i in idx if B[i][j] > b]
        return {"j": j, "b": b, "l": self._grow(L, B, g, depth - 1), "r": self._grow(R, B, g, depth - 1)}

    @staticmethod
    def _eval(t, brow):
        while "j" in t:
            t = t["l"] if brow[t["j"]] <= t["b"] else t["r"]
        return t["v"]

    def fit(self, X, y):
        self.edges = self._bin_edges(X)
        B = [self._binize(r) for r in X]
        self.base = sum(y) / len(y)
        pred = [self.base] * len(y)
        self.trees = []
        idx = list(range(len(y)))
        for _ in range(self.n):
            g = [yi - pi for yi, pi in zip(y, pred)]
            t = self._grow(idx, B, g, self.depth)
            self.trees.append(t)
            for i in idx:
                pred[i] += self.lr * self._eval(t, B[i])
        return self

    def predict_one(self, row):
        b = self._binize(row)
        return self.base + self.lr * sum(self._eval(t, b) for t in self.trees)

    def predict(self, X):
        return [self.predict_one(r) for r in X]


# --------------------------------------------------------------------------
# Linear & logistic regression
# --------------------------------------------------------------------------
def _solve(A, b):
    """Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


class LinearRegression:
    def __init__(self, l2=1e-3):
        self.l2 = l2

    def fit(self, X, y):
        Xb = [[1.0] + list(r) for r in X]
        d = len(Xb[0])
        A = [[sum(r[i] * r[j] for r in Xb) + (self.l2 if i == j and i else 0) for j in range(d)] for i in range(d)]
        b = [sum(r[i] * yi for r, yi in zip(Xb, y)) for i in range(d)]
        self.w = _solve(A, b)
        return self

    def predict_one(self, row):
        return self.w[0] + sum(w * x for w, x in zip(self.w[1:], row))

    def predict(self, X):
        return [self.predict_one(r) for r in X]


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


class LogisticRegression:
    def __init__(self, epochs=300, lr=0.3, l2=1e-3):
        self.epochs, self.lr, self.l2 = epochs, lr, l2

    def fit(self, X, y):
        d = len(X[0])
        self.mu = [sum(r[j] for r in X) / len(X) for j in range(d)]
        self.sd = [max(1e-9, math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in X) / len(X))) for j in range(d)]
        Z = [self._std(r) for r in X]
        self.w, self.b = [0.0] * d, 0.0
        n = len(Z)
        for _ in range(self.epochs):
            gw, gb = [0.0] * d, 0.0
            for z, yi in zip(Z, y):
                e = sigmoid(self.b + sum(w * v for w, v in zip(self.w, z))) - yi
                gb += e
                for j in range(d):
                    gw[j] += e * z[j]
            self.b -= self.lr * gb / n
            self.w = [w - self.lr * (g / n + self.l2 * w) for w, g in zip(self.w, gw)]
        return self

    def _std(self, r):
        return [(v - m) / s for v, m, s in zip(r, self.mu, self.sd)]

    def predict_proba_one(self, row):
        return sigmoid(self.b + sum(w * v for w, v in zip(self.w, self._std(row))))


# --------------------------------------------------------------------------
# Multilingual text classifier (multinomial naive Bayes on words + char n-grams)
# --------------------------------------------------------------------------
# Curated multilingual danger lexicon (EN / HI / KN, romanised + native script). A hit
# adds a shared "lex" feature so the model generalises to phrasings it has not seen.
DANGER_LEXICON = {
    "help", "ambulance", "emergency", "injured", "bleeding", "blood", "fire", "flood", "flooding", "trapped",
    "drowning", "unconscious", "breathing", "collapsed", "accident", "hurt", "hospital", "urgent", "shock",
    "madad", "bachao", "chot", "khoon", "behosh", "aag", "saans", "phas", "dard", "hosh",
    "sahaya", "kaapaadi", "gaaya", "raktha", "usiru", "benki", "thurtu", "sikki", "aspatre", "prajne",
    "मदद", "बचाओ", "एम्बुलेंस", "खून", "बेहोश", "आग", "सांस", "फंस", "दुर्घटना", "इमरजेंसी",
    "ಸಹಾಯ", "ಕಾಪಾಡಿ", "ಆಂಬುಲೆನ್ಸ್", "ಗಾಯ", "ಉಸಿರು", "ಬೆಂಕಿ", "ಸಿಕ್ಕಿ", "ಉಸಿರಾಟ",
}


def text_features(text):
    t = text.lower()
    words = re.findall(r"[^\s.,!?;:'\"()-]+", t)
    feats = ["w:" + w for w in words]
    feats += ["b:" + a + "_" + b for a, b in zip(words, words[1:])]
    s = " " + " ".join(words) + " "
    feats += ["c:" + s[i:i + 4] for i in range(len(s) - 3)]
    hits = sum(1 for w in words if any(w.startswith(k) for k in DANGER_LEXICON))
    feats += ["lex:hit"] * (4 * hits) or ["lex:none"] * 2
    return feats


class NaiveBayesText:
    def fit(self, texts, labels, alpha=0.5):
        self.alpha = alpha
        self.classes = sorted(set(labels))
        self.prior = {c: math.log(labels.count(c) / len(labels)) for c in self.classes}
        self.counts = {c: Counter() for c in self.classes}
        for t, c in zip(texts, labels):
            self.counts[c].update(text_features(t))
        self.vocab = set().union(*self.counts.values())
        self.totals = {c: sum(self.counts[c].values()) for c in self.classes}
        return self

    def log_scores(self, text):
        f = text_features(text)
        V = len(self.vocab)
        return {c: self.prior[c] + sum(math.log((self.counts[c][x] + self.alpha) / (self.totals[c] + self.alpha * V))
                                       for x in f if x in self.vocab) for c in self.classes}

    def proba(self, text, positive):
        s = self.log_scores(text)
        m = max(s.values())
        z = sum(math.exp(v - m) for v in s.values())
        return math.exp(s[positive] - m) / z

    def top_features(self, text, positive, k=4):
        f = [x for x in set(text_features(text)) if x in self.vocab and x.startswith("w:")]
        other = [c for c in self.classes if c != positive][0]
        V = len(self.vocab)

        def lr(x):
            return (math.log((self.counts[positive][x] + self.alpha) / (self.totals[positive] + self.alpha * V))
                    - math.log((self.counts[other][x] + self.alpha) / (self.totals[other] + self.alpha * V)))
        return [x[2:] for x in sorted(f, key=lr, reverse=True)[:k] if lr(x) > 0.5]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mae(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else 0.5


def binary_report(pred, truth):
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum((not p) and t for p, t in zip(pred, truth))
    return {"recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp)}


class BetaBandit:
    """Contextual Thompson-style bandit (posterior means) for connectivity path choice."""

    def __init__(self):
        self.a = defaultdict(lambda: 1.0)
        self.b = defaultdict(lambda: 1.0)

    def update(self, ctx, arm, success):
        if success:
            self.a[(ctx, arm)] += 1
        else:
            self.b[(ctx, arm)] += 1

    def mean(self, ctx, arm):
        a, b = self.a[(ctx, arm)], self.b[(ctx, arm)]
        return a / (a + b)

    def trials(self, ctx, arm):
        return self.a[(ctx, arm)] + self.b[(ctx, arm)] - 2
