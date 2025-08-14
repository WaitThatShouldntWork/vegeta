## Policy (v1, concise)

- Answer cost: risk_weight applied to low-confidence answers
- Ask cost: user_annoyance = 1.0
- Search cost: token_cost = 1.0; latency_weight = 0.001
- Risk score: 0.5*CVSS/10 + 0.4*EPSS + 0.1*KEV

Tunable in code; document deltas in commit messages.


