"""Check controlled paraphrases and abstention cases separately from the replay."""
from scope_grammar import select

cases = (
    ("the former instead of the latter gets accepted", 1),
    ("the latter as opposed to the former is given accepted", 1),
    ("accepted though withheld from the former is valid for the latter", 10),
    ("accepted despite a denial concerning the latter becomes effective for the former", 11),
    ("accepted was discussed with the former", -1),
    ("the former never receives accepted", -1),
    ("if the former receives accepted", -1),
    ("accepted applies to the former and pertains to the latter", -1),
    ("accepted applies to the former ; the latter receives accepted", -1),
    ("the former receives discarded", -1),
)

for sentence, expected in cases:
    actual = select(sentence.split())
    assert actual == expected, (sentence, expected, actual)
print(f"Passed {len(cases)} controlled scope checks.")
