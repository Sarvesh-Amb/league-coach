#!/usr/bin/env python3
"""League Coach: give it your match stats, get coaching back.
Rules engine finds your biggest gaps; a local open-weight LLM (via Ollama) explains them.
Zero dependencies (stdlib only). Usage:
    python league_coach.py --demo
    python league_coach.py my_games.json --model qwen2.5:7b
"""
import argparse, json, sys, urllib.request
from collections import defaultdict

# role: (CS/min target, vision score/min target, max avg deaths/game). Rough guides; edit freely.
BENCH = {"top": (6.5, 0.8, 5), "jungle": (5.5, 1.0, 5), "mid": (7.0, 0.9, 4.5),
         "adc": (7.5, 0.8, 4.5), "support": (1.2, 2.0, 5)}

DEMO = [
    {"role": "adc", "champion": "Jinx", "win": False, "kills": 4, "deaths": 8, "assists": 5, "cs": 150, "minutes": 28, "vision": 14},
    {"role": "adc", "champion": "Ashe", "win": True, "kills": 9, "deaths": 3, "assists": 10, "cs": 210, "minutes": 31, "vision": 22},
    {"role": "adc", "champion": "Jinx", "win": False, "kills": 3, "deaths": 9, "assists": 4, "cs": 140, "minutes": 27, "vision": 12},
    {"role": "adc", "champion": "Kai'Sa", "win": True, "kills": 7, "deaths": 4, "assists": 8, "cs": 195, "minutes": 30, "vision": 20},
    {"role": "adc", "champion": "Jinx", "win": False, "kills": 2, "deaths": 7, "assists": 6, "cs": 160, "minutes": 29, "vision": 15},
    {"role": "adc", "champion": "Ashe", "win": False, "kills": 5, "deaths": 6, "assists": 7, "cs": 175, "minutes": 33, "vision": 18},
]

SYSTEM = ("You are a blunt but encouraging League of Legends coach. Use ONLY the numbers provided; never invent stats. "
          "Reply with: 1) the single biggest problem, 2) three concrete drills for the next 5 games, "
          "3) one thing the player does well. Keep it under 250 words.")


def avg(xs):
    return sum(xs) / len(xs) if xs else 0


def analyze(games):
    by = defaultdict(list)
    for g in games:
        by[g["role"].lower()].append(g)
    report = {}
    for role, gs in by.items():
        n, mins = len(gs), sum(g["minutes"] for g in gs)
        wins = [g for g in gs if g["win"]]
        losses = [g for g in gs if not g["win"]]
        s = {
            "games": n,
            "winrate_pct": round(100 * len(wins) / n),
            "cs_per_min": round(sum(g["cs"] for g in gs) / mins, 2),
            "vision_per_min": round(sum(g["vision"] for g in gs) / mins, 2),
            "avg_deaths": round(avg([g["deaths"] for g in gs]), 1),
            "deaths_in_wins": round(avg([g["deaths"] for g in wins]), 1),
            "deaths_in_losses": round(avg([g["deaths"] for g in losses]), 1),
            "kda": round(sum(g["kills"] + g["assists"] for g in gs) / max(1, sum(g["deaths"] for g in gs)), 2),
        }
        cs_t, vis_t, d_t = BENCH.get(role, BENCH["mid"])
        f = []  # (severity, message)
        if s["cs_per_min"] < cs_t * 0.92:
            f.append(((cs_t - s["cs_per_min"]) / cs_t, f"CS/min {s['cs_per_min']} vs ~{cs_t} target for {role}"))
        if s["vision_per_min"] < vis_t * 0.85:
            f.append(((vis_t - s["vision_per_min"]) / vis_t, f"Vision/min {s['vision_per_min']} vs ~{vis_t} target"))
        if s["avg_deaths"] > d_t:
            f.append(((s["avg_deaths"] - d_t) / d_t, f"Avg deaths {s['avg_deaths']} vs <= {d_t} target"))
        if losses and wins and s["deaths_in_losses"] - s["deaths_in_wins"] >= 2:
            f.append((0.5, f"You die {s['deaths_in_losses']}x in losses vs {s['deaths_in_wins']}x in wins: your losses are mostly feeding games"))
        s["findings"] = [m for _, m in sorted(f, reverse=True)] or ["No major gaps vs benchmarks. Keep it up."]
        report[role] = s
    return report


def ask_llm(report, model, host):
    body = json.dumps({"model": model, "stream": False, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "My stats and flagged gaps:\n" + json.dumps(report, indent=2)}]}).encode()
    req = urllib.request.Request(host + "/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["message"]["content"]


def main():
    ap = argparse.ArgumentParser(description="Stats-driven League of Legends coach")
    ap.add_argument("stats", nargs="?", help="JSON file: list of games (see DEMO in this file)")
    ap.add_argument("--demo", action="store_true", help="use built-in sample games")
    ap.add_argument("--model", default="llama3.1:8b", help="any model you've pulled in Ollama")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--no-llm", action="store_true", help="rules-only output")
    a = ap.parse_args()
    if not a.demo and not a.stats:
        ap.error("give a stats file or --demo")
    games = DEMO if a.demo else json.load(open(a.stats))
    report = analyze(games)
    for role, s in report.items():
        print(f"\n== {role.upper()} ({s['games']} games, {s['winrate_pct']}% WR, {s['kda']} KDA) ==")
        for m in s["findings"]:
            print(" -", m)
    if a.no_llm:
        return
    print(f"\n== COACH ({a.model}) ==")
    try:
        print(ask_llm(report, a.model, a.host))
    except Exception as e:
        print(f"(Ollama not reachable: {e}. Start it with `ollama serve` and `ollama pull {a.model}`.)")


if __name__ == "__main__":
    sys.exit(main())
