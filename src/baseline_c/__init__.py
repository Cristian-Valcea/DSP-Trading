"""
Baseline C: Multi-Time Supervised Learning

Multi-horizon supervised strategy with 4 rebalance times:
- 10:31 -> 11:31
- 11:31 -> 12:31
- 12:31 -> 14:00
- 14:00 -> next day 10:31 (overnight)

Key differences from Baseline B:
- Multiple decision times per day (not just 10:31)
- Re-entry allowed
- No forced daily flatten (positions can be held overnight)
- 4 separate Ridge models (one per interval)
"""
