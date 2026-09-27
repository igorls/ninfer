"""Create a small, deterministic diagnostic cohort with executable gold labels.

These authored toy tasks are not representative capability benchmarks. The gold label and
oracle explanation remain outside the state/question sent to the model.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path


def make_cases(seed=20260923, scenarios=4):
    rng = random.Random(seed)
    rows = []

    def add(family, index, state, question, labels, expected, oracle):
        ident = f"oracle-{seed}-{family}-{index:03d}"
        rows.append({"id": ident, "family": family, "group": ident, "split": "private",
                     "state": state, "question": question, "labels": labels, "expected": expected,
                     "provenance": {"source": "NInfer authored exact-oracle diagnostic", "seed": seed,
                                    "oracle": oracle, "license": "MIT"}})

    for i in range(scenarios):
        # Integer arithmetic: fee, deduction and cap applied in the specified order.
        units, price, fee, deduction = rng.randrange(20, 90), rng.randrange(7, 25), rng.randrange(10, 60), rng.randrange(30, 100)
        subtotal = units * price + fee - deduction
        cap = subtotal + ([-17, 23, -41, 57][i % 4])
        paid = min(subtotal, cap)
        numbers = list(dict.fromkeys([paid, units * price - deduction, min(units * price, cap) + fee - deduction, cap + fee]))
        while len(numbers) < 4:
            candidate = paid + rng.randrange(1, 80)
            if candidate not in numbers:
                numbers.append(candidate)
        rng.shuffle(numbers)
        criteria = {f"amount_{n}": f"Pay exactly {n} credits." for n in numbers}
        labels = list(criteria)
        add("integer_accounting", i,
            f"Shipment: {units} units at {price} credits per unit. Add a single fee of {fee} credits, then subtract the {deduction}-credit deduction, then cap the final payment at {cap} credits. A clerk's preliminary note says to apply the cap before the fee; that note is not the rule.",
            {"type": "choice", "instructions": "Which final payment follows the rule?", "criteria": criteria},
            labels, f"amount_{paid}", f"min({units}*{price}+{fee}-{deduction}, {cap}) = {paid}")

        start = date(2026, rng.randrange(1, 10), rng.randrange(1, 20))
        duration = rng.randrange(12, 45)
        offset = [-1, 0, 1, 2][i % 4]
        submitted = start + timedelta(days=duration - 1 + offset)
        deadline = start + timedelta(days=duration - 1)
        add("inclusive_dates", i,
            f"The filing window lasts {duration} calendar days. {start.isoformat()} is day 1, and filing on the final day is allowed. No weekend or holiday extension applies. The request arrived on {submitted.isoformat()}. A draft note counted the start day as day 0.",
            {"type": "noul", "instructions": "Was this request filed within the allowed window?", "criteria": {"true": "Filed on or before the inclusive last day.", "false": "Filed after the inclusive last day."}},
            ["no", "yes"], "yes" if submitted <= deadline else "no",
            f"last_day = start + {duration}-1 days = {deadline.isoformat()}")

        # Cover each decision and both sides of the waiver exception in the default cohort.
        active, prohibited, exception = [
            (False, False, True), (True, False, False),
            (True, True, False), (True, True, True),
        ][i % 4]
        approved = active and (not prohibited or exception)
        add("rule_precedence", i,
            "Dispatch rules: an inactive account must always be rejected. For an active account, a restricted destination is rejected unless it has a signed waiver; a waiver overrides only the destination restriction. An unrestricted destination needs no waiver.\n"
            f"Account active: {str(active).lower()}. Destination restricted: {str(prohibited).lower()}. Signed waiver present: {str(exception).lower()}.\nA colleague says a waiver overrides every rule; the rules above are authoritative.",
            {"type": "choice", "instructions": "Choose the permitted dispatch decision.", "criteria": {"dispatch": "Dispatch the package.", "reject_account": "Reject because the account is inactive.", "reject_destination": "Reject because the restricted destination has no applicable waiver."}},
            ["dispatch", "reject_account", "reject_destination"],
            "dispatch" if approved else "reject_account" if not active else "reject_destination",
            f"active={active}; restricted={prohibited}; waiver={exception}; active AND (NOT restricted OR waiver)={approved}")

        names = ["Amber", "Birch", "Cedar", "Dune"]
        rng.shuffle(names)
        owners = dict(zip(names, ["Iris", "Noel", "Uma", "Zane"]))
        source = names[i % 4]
        transfer_to = names[(i + 1) % 4]
        renamed = names[(i + 2) % 4]
        answer = owners[transfer_to]
        lines = [f"Original queue {name} belongs to {owner}." for name, owner in owners.items()]
        rng.shuffle(lines)
        add("temporal_lookup", i,
            "\n".join(lines) + f"\nTicket X started in queue {source}. At 09:10 it was moved to queue {transfer_to}. At 09:15 queue {renamed} changed owner to Petra. At 09:20 an obsolete dashboard still displayed X under {source}. Ticket ownership follows its current queue; a stale display does not move a ticket.",
            {"type": "choice", "instructions": "Who owns ticket X at 09:25?", "criteria": {n: n for n in ["Iris", "Noel", "Uma", "Zane", "Petra"]}},
            ["Iris", "Noel", "Uma", "Zane", "Petra"], answer,
            f"X current queue={transfer_to}; unrelated changed queue={renamed}; owner={answer}")

        errors, outage, loss = [0, 7, 12, 2][i % 4], i % 4 == 2, i % 4 == 3
        severity = 3 if loss else 2 if outage else 1 if errors >= 5 else 0
        add("ordinal_rules", i,
            f"Incident facts: {errors} affected requests; complete service outage={str(outage).lower()}; irreversible data loss={str(loss).lower()}. A draft severity label was assigned before these facts were checked and must be ignored.",
            {"type": "score", "instructions": "Assign the highest applicable level; higher levels override lower ones.", "criteria": ["No outage, no data loss, and fewer than five affected requests.", "At least five affected requests, with no outage or data loss.", "Complete service outage with no data loss, regardless of request count.", "Irreversible data loss, regardless of outage or request count."]},
            ["0", "1", "2", "3"], severity,
            "3 if loss else 2 if outage else 1 if errors >= 5 else 0")

        available, requested, reserve = rng.randrange(80, 130), rng.randrange(20, 60), rng.randrange(10, 25)
        held = available - requested - reserve + [-1, 0, 1, 7][i % 4]
        free_after = available - held - requested
        add("resource_constraints", i,
            f"A pool contains {available} units in total. Existing holds reserve {held} of those units, and held units cannot be allocated again. A new request requires {requested} units. After honoring holds and the new request, at least {reserve} units must remain free; equality is permitted. Pending releases are not available yet.",
            {"type": "noul", "instructions": "Can the new request be approved under the reserve rule?", "criteria": {"true": "The remaining free capacity meets the minimum reserve.", "false": "The remaining free capacity falls below the minimum reserve."}},
            ["no", "yes"], "yes" if free_after >= reserve else "no",
            f"{available}-{held}-{requested} = {free_after}; required >= {reserve}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--scenarios-per-family", type=int, default=4)
    args = parser.parse_args()
    if args.scenarios_per_family < 1:
        parser.error("scenarios-per-family must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        for row in make_cases(args.seed, args.scenarios_per_family):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
