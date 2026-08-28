# Claim relationship labeling guide

This guide separates the repository's Task 8 claim relationships from SciFact's
native three-way labels. It is annotation guidance, not a conversion table.

## Task 8 evidence relationships

- `entails`: the cited passage establishes every material part of the atomic
  claim, including quantities, direction, population, method, and qualifiers.
- `partial`: the passage establishes a meaningful strict subset of the same
  claim without contradicting the remainder. For example, evidence that a
  method improved accuracy but gives no magnitude is partial support for “the
  method improved accuracy by 10 percentage points.”
- `does_not_support`: the passage is unrelated, merely topical, contradicts a
  material part, or establishes none of the claim. Reasons should distinguish
  contradiction from missing information even though both share this structural
  relationship value.

Annotators should first split compound assertions into minimal claims. `partial`
is not a substitute for leaving two separable assertions combined.

## SciFact labels

- `SUPPORT`: at least one rationale sentence set establishes the full claim.
- `CONTRADICT`: at least one rationale sentence set directly refutes the claim.
- `NOT_ENOUGH_INFO`: the supplied cited documents neither establish nor directly
  refute the full claim. A topical passage or a missing qualifier is not, by
  itself, contradiction.

SciFact has no native `partial` label. Therefore SciFact label accuracy must be
reported separately from Task 8 relationship accuracy. The external runner uses
the cited documents supplied by SciFact and measures three-way classification
plus rationale selection; it does not evaluate repository retrieval or the
LangGraph flow.
