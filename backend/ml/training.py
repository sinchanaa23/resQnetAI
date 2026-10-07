"""Synthetic data generation, training and evaluation for every resQnet model.

All training data comes from the simulator (proposal §5: "Synthetic training data").
Each model is evaluated against the rule baseline so the dashboard can show
exactly how much the AI improves on it.
"""
import math
import random
import time

from ..sim.roads import CLASS_RANK, FREE_SPEED, _peakness
from .core import (BetaBandit, GBMRegressor, LinearRegression, LogisticRegression, NaiveBayesText, auc,
                   binary_report, mae, sigmoid)

# --------------------------------------------------------------------------
# Multilingual emergency / non-emergency phrases (English, Hindi, Kannada —
# romanised and native script) used for the call-prompt classifier.
# --------------------------------------------------------------------------
EMERG_CORE = [
    "help please water is entering my house", "my house is flooding", "we are trapped on the roof",
    "someone is injured badly", "there has been an accident", "my father collapsed and is not responding",
    "need an ambulance urgently", "child is not breathing", "there is a fire in the building",
    "he got an electric shock", "she is bleeding a lot", "a man is drowning in the drain",
    "the wall fell on people", "my mother is having chest pain", "car stuck in water people inside",
    "old man fell unconscious", "pregnant woman needs hospital now", "house collapsed people trapped",
    "madad karo paani ghar mein ghus gaya", "bachao hum phas gaye hain", "ambulance chahiye jaldi",
    "chot lagi hai khoon beh raha hai", "saans nahi le raha hai", "behosh ho gaya hai", "aag lagi hai",
    "accident ho gaya hai madad chahiye", "ghar mein paani bhar gaya bachao", "dil ka daura pada hai",
    "मदद करो पानी घर में घुस गया", "बचाओ हम फंस गए हैं", "एम्बुलेंस चाहिए जल्दी", "खून बह रहा है",
    "बेहोश हो गया है", "आग लगी है", "सांस नहीं ले रहा", "दुर्घटना हो गई है",
    "sahaya maadi manege neeru nuggide", "kaapaadi naavu sikkikondiddeve", "ambulance beku bega",
    "gaaya aagide raktha hogtide", "usiru aadtilla", "prajne tappide", "benki hattide",
    "ಸಹಾಯ ಮಾಡಿ ಮನೆಗೆ ನೀರು ನುಗ್ಗಿದೆ", "ಕಾಪಾಡಿ", "ಆಂಬುಲೆನ್ಸ್ ಬೇಕು", "ಗಾಯ ಆಗಿದೆ", "ಉಸಿರು ಆಡುತ್ತಿಲ್ಲ", "ಬೆಂಕಿ ಹತ್ತಿದೆ",
    "yes emergency", "yes i need help", "haan emergency hai", "haudu thurtu", "हाँ इमरजेंसी है",
    "please send ambulance he is not breathing", "water rising fast we cannot get out", "my son is unconscious",
    "there is blood everywhere", "someone fell into the flood water", "power line fell on the road someone hurt",
    "usiru kashta aagtide", "ambulance kalsi", "neeru jaasti aagtide sikkikondidivi", "aspatrege kar kondu hogi",
    "madad bhejo ambulance", "paani bahut badh gaya nikal nahi pa rahe", "khoon nikal raha hai", "hosh nahi hai",
    "एम्बुलेंस भेजो", "पानी बढ़ रहा है हम फंसे हैं", "सांस लेने में तकलीफ", "ಉಸಿರಾಟದ ತೊಂದರೆ", "ಆಂಬುಲೆನ್ಸ್ ಕಳಿಸಿ",
    "ನೀರು ಹೆಚ್ಚಾಗುತ್ತಿದೆ ಸಿಕ್ಕಿಕೊಂಡಿದ್ದೇವೆ", "accident aagide gaaya aagide", "thurtu sahaya beku",
]
NONEMERG_CORE = [
    "no i am fine just calling my friend", "all safe here thank you", "calling about the office meeting",
    "checking my order delivery", "i will reach home late", "no emergency just checking on mom",
    "is the road open to go to office", "no need we are okay", "wrong number sorry", "calling the bank",
    "just wanted to know if power will come back", "my friend is fine now no need",
    "is there flooding on the road i need to go to work", "please call me later",
    "nahi sab theek hai", "kuch nahi bas aise hi call kiya", "hum safe hain", "office ka kaam hai",
    "baad mein baat karte hain", "nahi emergency nahi hai", "ghar pahunch gaya hoon",
    "मैं ठीक हूँ", "सब ठीक है", "बाद में बात करते हैं", "नहीं इमरजेंसी नहीं है",
    "illa naavu chennagiddeve", "yella chennagide", "office kelasa", "aamele maathaadthini",
    "chinte illa naavu safe", "illa emergency illa",
    "ನಾವು ಸುರಕ್ಷಿತ", "ಎಲ್ಲಾ ಚೆನ್ನಾಗಿದೆ", "ಆಮೇಲೆ ಮಾತಾಡ್ತೀನಿ", "no",
    "just calling to say happy birthday", "when will the network be normal", "is school closed tomorrow",
    "booking a cab to airport", "i am at home everything normal", "food delivery is late", "talking to my wife",
    "kal milte hain", "khana kha liya", "school band hai kya", "naale sigona", "oota aaytha",
    "कल मिलते हैं", "खाना खा लिया", "ನಾಳೆ ಸಿಗೋಣ", "ಊಟ ಆಯ್ತಾ", "nothing serious all good",
]
PREFIX = ["", "", "hello", "please", "sir", "jaldi", "bega", "hello madam", "suniye", "ri"]
SUFFIX = ["", "", "near ejipura", "in koramangala", "at 6th block", "hsr layout", "please come fast",
          "jaldi aao", "bega banni", "near st johns", "on 80 feet road", "sir"]


TYPO = str.maketrans({"a": "aa", "e": "i"})


def gen_texts(rng, n, emerg=EMERG_CORE, normal=NONEMERG_CORE):
    texts, labels = [], []
    for _ in range(n):
        e = rng.random() < 0.5
        core = rng.choice(emerg if e else normal)
        if rng.random() < 0.15:  # speech-to-text / spelling noise
            w = core.split()
            i = rng.randrange(len(w))
            w[i] = w[i].translate(TYPO)
            core = " ".join(w)
        t = " ".join(x for x in (rng.choice(PREFIX), core, rng.choice(SUFFIX)) if x)
        texts.append(t)
        labels.append("emergency" if e else "normal")
    return texts, labels


# --------------------------------------------------------------------------
# Hidden "ground truth" generators for the other models
# --------------------------------------------------------------------------
def true_accept_prob(f):
    dist, night, rate, assign, spec, signal = f
    return sigmoid(1.6 - 1.0 * dist - 0.9 * night + 3.2 * (rate - 0.5) - 1.6 * assign + 0.9 * spec + 0.8 * (signal - 0.5))


def severity_features(inc_type, people, injuries, critical, in_zone):
    t = {"flood": 3, "fire": 4, "accident": 3, "cardiac": 5, "collapse": 4, "other": 1}.get(inc_type, 2)
    return [t, math.log1p(people), min(injuries, 10), 1 if critical else 0, 1 if in_zone else 0]


def expert_severity(f, rng):
    t, lp, inj, crit, zone = f
    s = 0.7 + 0.45 * t + 0.35 * lp + 0.18 * inj + 1.3 * crit + 0.35 * zone + rng.gauss(0, 0.3)
    return max(1, min(5, round(s)))


def rule_severity(f):
    t, lp, inj, crit, zone = f
    s = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}[t] + crit + (1 if inj >= 3 else 0) + (1 if lp > 2.4 else 0)
    return max(1, min(5, s))


PATHS = ["cellular", "neighbour_cell", "temp_cell", "satellite", "d2d_relay"]
PATH_LATENCY = {"cellular": 0.35, "neighbour_cell": 0.6, "temp_cell": 0.5, "satellite": 6.0, "d2d_relay": 14.0}


def path_success_prob(ctx, arm):
    home, slice_on, nb, cow, sat, d2d = ctx
    if arm == "cellular":
        return {"up": 0.99, "congested": 0.995 if slice_on else 0.78, "down": 0.0}[home]
    if arm == "neighbour_cell":
        return (0.93 if home != "congested" or slice_on else 0.83) if nb else 0.0
    if arm == "temp_cell":
        return 0.97 if cow else 0.0
    if arm == "satellite":
        return 0.9 if sat else 0.0
    return 0.86 if d2d else 0.0


def rule_path(ctx):
    home, slice_on, nb, cow, sat, d2d = ctx
    avail = [home != "down", nb, cow, sat, d2d]
    for arm, ok in zip(PATHS, avail):
        if ok:
            return arm
    return None


def random_ctx(rng):
    home = rng.choice(["up", "congested", "congested", "down"])
    return (home, rng.random() < 0.5, rng.random() < 0.7, rng.random() < 0.4, rng.random() < 0.3, rng.random() < 0.6)


class ModelSuite:
    """Trains every model at startup (~1 s) and keeps the evaluation report."""

    def __init__(self, graph, seed=7):
        self.graph = graph
        self.rng = random.Random(seed)
        self.report = {}
        t0 = time.perf_counter()
        self._train_eta()
        self._train_inflow()
        self._train_acceptance()
        self._train_classifier()
        self._train_severity()
        self._train_bandit()
        self.train_seconds = round(time.perf_counter() - t0, 2)

    # ---- F2: segment ETA ------------------------------------------------
    def eta_features(self, ei, hour, probe_kmh):
        g = self.graph
        flood = g.flood_factor(ei)
        return [CLASS_RANK[g.cls[ei]], g.length[ei], _peakness(hour), flood, probe_kmh if probe_kmh else -1.0]

    def observe_probe(self, ei, hour, rng=None):
        """Noisy traffic-probe observation (like aggregated floating-car data); missing 30% of the time."""
        r = rng or self.rng
        if r.random() < 0.3:
            return None
        return self.graph.true_speed_kmh(ei, hour) * math.exp(r.gauss(0, 0.25))

    def _train_eta(self):
        g, r = self.graph, self.rng
        center = ((g.bbox[0] + g.bbox[2]) / 2, (g.bbox[1] + g.bbox[3]) / 2)
        X, y, base, lens = [], [], [], []
        saved = g.flood
        for k in range(2600):
            g.flood = (center, 900) if k % 3 == 0 else None
            ei = r.randrange(len(g.length))
            hour = r.uniform(6, 23)
            X.append(self.eta_features(ei, hour, self.observe_probe(ei, hour)))
            y.append(g.true_time(ei, hour) / max(1.0, g.length[ei]))  # pace (s / m)
            base.append(g.free_time(ei))
            lens.append(g.length[ei])
        g.flood = saved
        n = 2000
        self.eta = GBMRegressor(n_estimators=70, max_depth=3).fit(X[:n], y[:n])
        pred = [p * L for p, L in zip(self.eta.predict(X[n:]), lens[n:])]
        truth = [t * L for t, L in zip(y[n:], lens[n:])]
        b_mae, m_mae = mae(base[n:], truth), mae(pred, truth)
        self.report["eta"] = {
            "feature": "F2", "task": "Segment ETA prediction", "model": "Gradient-boosted trees (70 x depth-3)",
            "metric": "MAE (s / segment)", "baseline_name": "Free-flow shortest-path",
            "baseline": round(b_mae, 2), "ai": round(m_mae, 2), "lower_is_better": True,
            "improvement_pct": round(100 * (b_mae - m_mae) / b_mae, 1)}

    def predict_edge_time(self, ei, hour, probe_kmh):
        pace = max(0.02, self.eta.predict_one(self.eta_features(ei, hour, probe_kmh)))
        return pace * max(1.0, self.graph.length[ei])

    # ---- F2: crowd inflow ----------------------------------------------
    @staticmethod
    def inflow_series(rng, base, days=3, step_min=5):
        out, ar = [], 0.0
        for k in range(days * 24 * 60 // step_min):
            hour = (k * step_min / 60) % 24
            ar = 0.9 * ar + rng.gauss(0, 0.05)
            # Aggregated cell counts carry sampling noise on top of the real trend.
            out.append(max(0.0, base * (0.25 + _peakness(hour)) * (1 + ar) * (1 + rng.gauss(0, 0.18))))
        return out

    @staticmethod
    def inflow_feats(hist, hour):
        return [hist[-1], hist[-2], hist[-3], sum(hist[-6:]) / 6, math.sin(2 * math.pi * hour / 24),
                math.cos(2 * math.pi * hour / 24)]

    def _train_inflow(self):
        r = self.rng
        X, y, Xt, yt, pt = [], [], [], [], []
        for c in range(24):
            s = self.inflow_series(r, r.uniform(40, 220))
            split = int(len(s) * 0.75)
            for k in range(6, len(s) - 2):
                hour = (k * 5 / 60) % 24
                f = self.inflow_feats(s[:k], hour)
                target = s[k] + s[k + 1]  # people entering in the next 10 minutes
                if k < split:
                    X.append(f)
                    y.append(target)
                else:
                    Xt.append(f)
                    yt.append(target)
                    pt.append(2 * s[k - 1])  # persistence baseline
        self.inflow = LinearRegression().fit(X, y)
        m = mae(self.inflow.predict(Xt), yt)
        b = mae(pt, yt)
        self.report["inflow"] = {
            "feature": "F2", "task": "Crowd inflow forecast (next 10 min)", "model": "Autoregressive + time-of-day",
            "metric": "MAE (people / cell)", "baseline_name": "Persistence (last value)",
            "baseline": round(b, 2), "ai": round(m, 2), "lower_is_better": True,
            "improvement_pct": round(100 * (b - m) / b, 1)}

    def predict_inflow(self, hist, hour):
        return max(0.0, self.inflow.predict_one(self.inflow_feats(hist, hour)))

    # ---- F3: acceptance probability ------------------------------------
    def _train_acceptance(self):
        r = self.rng
        X, y = [], []
        for _ in range(2400):
            f = [r.uniform(0.1, 5), 1 if r.random() < 0.3 else 0, r.uniform(0.2, 0.95), r.choice([0, 0, 0, 1]),
                 1 if r.random() < 0.6 else 0, r.uniform(0.2, 1.0)]
            X.append(f)
            y.append(1 if r.random() < true_accept_prob(f) else 0)
        n = 1800
        self.accept = LogisticRegression().fit(X[:n], y[:n])
        ai_auc = auc([self.accept.predict_proba_one(f) for f in X[n:]], y[n:])
        base_auc = auc([-f[0] for f in X[n:]], y[n:])
        # Escalation rounds: nearest-first (baseline) vs AI expected-utility ranking.
        rounds_b, rounds_ai = [], []
        for _ in range(400):
            cands = [[r.uniform(0.2, 4.5), 1 if r.random() < 0.3 else 0, r.uniform(0.2, 0.95),
                      r.choice([0, 0, 0, 1]), 1 if r.random() < 0.6 else 0, r.uniform(0.2, 1.0)] for _ in range(8)]
            outcomes = [r.random() < true_accept_prob(c) for c in cands]
            for order, sink in ((sorted(range(8), key=lambda i: cands[i][0]), rounds_b),
                                (sorted(range(8), key=lambda i: -self.match_score(cands[i])[0]), rounds_ai)):
                k = next((j + 1 for j, i in enumerate(order) if outcomes[i]), 8)
                sink.append(k)
        rb, ra = sum(rounds_b) / len(rounds_b), sum(rounds_ai) / len(rounds_ai)
        self.report["acceptance"] = {
            "feature": "F3", "task": "Volunteer acceptance probability", "model": "Logistic regression",
            "metric": "AUC", "baseline_name": "Distance-only heuristic",
            "baseline": round(base_auc, 3), "ai": round(ai_auc, 3), "lower_is_better": False,
            "improvement_pct": round(100 * (ai_auc - base_auc) / base_auc, 1)}
        self.report["ranking"] = {
            "feature": "F3", "task": "Responder ranking", "model": "Expected-utility ranking (p_accept x fit / ETA)",
            "metric": "Avg. escalation rounds", "baseline_name": "Nearest-first",
            "baseline": round(rb, 2), "ai": round(ra, 2), "lower_is_better": True,
            "improvement_pct": round(100 * (rb - ra) / rb, 1)}

    def match_score(self, f):
        """Returns (score, p_accept, eta_min). f = [dist_km, night, rate, assign, spec, signal]."""
        p = self.accept.predict_proba_one(f)
        eta = 2.0 + f[0] / 0.35  # ~21 km/h urban average for a responder on a two-wheeler + 2 min prep
        fit = 1.0 + 0.6 * f[4]
        return p * fit / (eta + 3), p, eta

    # ---- F4: emergency call / SMS classifier ---------------------------
    def _train_classifier(self):
        # Evaluate on phrasings the model never saw (every 5th core phrase is held out),
        # then retrain on everything for live use.
        r = random.Random(11)
        e_tr = [p for i, p in enumerate(EMERG_CORE) if i % 5]
        n_tr = [p for i, p in enumerate(NONEMERG_CORE) if i % 5]
        e_te = [p for i, p in enumerate(EMERG_CORE) if not i % 5]
        n_te = [p for i, p in enumerate(NONEMERG_CORE) if not i % 5]
        tr_x, tr_y = gen_texts(r, 1200, e_tr, n_tr)
        te_x, te_y = gen_texts(r, 600, e_te, n_te)
        self.clf_threshold = 0.35  # biased for recall: uncertain cases are treated as emergencies
        held = NaiveBayesText().fit(tr_x, tr_y)
        pred = [held.proba(t, "emergency") >= self.clf_threshold for t in te_x]
        truth = [l == "emergency" for l in te_y]
        rep = binary_report(pred, truth)
        kw = ["help", "madad", "bachao", "ambulance", "sahaya", "kaapaadi", "emergency"]
        bpred = [any(k in t.lower() for k in kw) for t in te_x]
        brep = binary_report(bpred, truth)
        self.report["classifier"] = {
            "feature": "F4", "task": "Emergency call/SMS classification (EN / HI / KN)",
            "model": "Multinomial naive Bayes, word + char n-grams", "metric": "Emergency recall",
            "baseline_name": "Keyword match", "baseline": round(brep["recall"], 3), "ai": round(rep["recall"], 3),
            "precision": round(rep["precision"], 3), "baseline_precision": round(brep["precision"], 3),
            "lower_is_better": False, "note": "evaluated on held-out phrasings",
            "improvement_pct": round(100 * (rep["recall"] - brep["recall"]) / max(0.01, brep["recall"]), 1)}
        all_x, all_y = gen_texts(r, 1600)
        self.clf = NaiveBayesText().fit(all_x, all_y)

    def classify(self, text):
        t = text.strip()
        if t in ("1", "one", "ek", "ondu"):
            return {"label": "emergency", "p": 1.0, "why": ["keypad 1"]}
        if t in ("2",):
            return {"label": "normal", "p": 0.0, "why": ["keypad 2"]}
        p = self.clf.proba(t, "emergency")
        lab = "emergency" if p >= self.clf_threshold else "normal"
        why = self.clf.top_features(t, "emergency" if lab == "emergency" else "normal")
        return {"label": lab, "p": round(p, 3), "why": why, "uncertain": 0.35 <= p < 0.65}

    # ---- Orchestrator: incident severity -------------------------------
    def _train_severity(self):
        r = self.rng
        types = ["flood", "fire", "accident", "cardiac", "collapse", "other"]
        X, y = [], []
        for _ in range(1500):
            f = severity_features(r.choice(types), int(r.expovariate(1 / 6)), r.choice([0, 0, 1, 1, 2, 3, 5, 8]),
                                  r.random() < 0.25, r.random() < 0.4)
            X.append(f)
            y.append(expert_severity(f, r))
        n = 1100
        self.sev = GBMRegressor(n_estimators=50, max_depth=3, min_leaf=10).fit(X[:n], y[:n])
        ai = sum(max(1, min(5, round(self.sev.predict_one(f)))) == t for f, t in zip(X[n:], y[n:])) / (len(X) - n)
        base = sum(rule_severity(f) == t for f, t in zip(X[n:], y[n:])) / (len(X) - n)
        self.report["severity"] = {
            "feature": "All", "task": "Incident severity (1-5)", "model": "Rule + gradient-boosting hybrid",
            "metric": "Agreement with expert labels", "baseline_name": "Rule table",
            "baseline": round(base, 3), "ai": round(ai, 3), "lower_is_better": False,
            "improvement_pct": round(100 * (ai - base) / max(0.01, base), 1)}

    def severity(self, inc_type, people, injuries, critical, in_zone):
        f = severity_features(inc_type, people, injuries, critical, in_zone)
        gbm = self.sev.predict_one(f)
        rule = rule_severity(f)
        s = max(1, min(5, round(0.75 * gbm + 0.25 * rule)))
        if critical:
            s = max(s, 4)  # safety floor: life-threatening reports are never below 4
        return s

    # ---- F5: connectivity path selection --------------------------------
    def _train_bandit(self):
        r = self.rng
        self.bandit = BetaBandit()
        for _ in range(60000):
            ctx = random_ctx(r)
            arm = r.choice(PATHS)
            p = path_success_prob(ctx, arm)
            self.bandit.update(ctx, arm, r.random() < p)
        res = {"rule": [0, 0.0], "ai": [0, 0.0]}
        N = 3000
        for _ in range(N):
            ctx = random_ctx(r)
            for name, arm in (("rule", rule_path(ctx)), ("ai", self.choose_paths(ctx)[0] if self.choose_paths(ctx) else None)):
                if arm and r.random() < path_success_prob(ctx, arm):
                    res[name][0] += 1
                    res[name][1] += PATH_LATENCY[arm]
        b, a = res["rule"][0] / N, res["ai"][0] / N
        self.report["paths"] = {
            "feature": "F5", "task": "Connectivity path selection", "model": "Contextual bandit (Beta posteriors)",
            "metric": "First-attempt delivery success", "baseline_name": "Fixed fallback order",
            "baseline": round(b, 3), "ai": round(a, 3), "lower_is_better": False,
            "improvement_pct": round(100 * (a - b) / b, 1)}

    def choose_paths(self, ctx):
        """Ranked list of usable paths for this context, best first."""
        home, slice_on, nb, cow, sat, d2d = ctx
        avail = dict(zip(PATHS, [home != "down", nb, cow, sat, d2d]))
        scored = [(self.bandit.mean(ctx, a) - 0.01 * PATH_LATENCY[a], a) for a in PATHS if avail[a]]
        return [a for _, a in sorted(scored, reverse=True)]
