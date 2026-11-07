"""
Dolphin network 4-subcommunity ground truth, reconstructed from Figure 1 of
Lusseau & Newman (2004), "Identifying the role that individual animals play
in their social network" (arXiv:q-bio/0403029) -- visually read off the
figure's four vertex shadings (white/dark-grey/black/light-grey), matching
the paper's own cited group sizes (23, 20, 12, 7 -> total 62, matching
TPOC-LPA's Table 4 "4 ground communities" for Dolphin).
"""

DOLPHIN_GROUPS = {
    0: [  # white, n=23
        "Fork", "TR120", "Stripes", "TR88", "Shmuddel", "TSN83", "TSN103",
        "Scabs", "SN63", "Whitetip", "Zap", "SN9", "SN4", "Grin", "Hook",
        "Kringel", "CCL", "Double", "Zipfel", "Thumper", "SN89", "SN100", "TR99",
    ],
    1: [  # black, n=20
        "DN16", "Wave", "Web", "SN90", "Gallatin", "Feather", "Upbang", "DN21",
        "Jet", "Number1", "Beescratch", "DN63", "Mus", "Notch", "Knit",
        "Ripplefluke", "Zig", "MN23", "Quasi", "TR82",
    ],
    2: [  # dark grey, n=12
        "Cross", "Five", "SMN5", "Trigger", "MN83", "Patchback", "Vau",
        "Haecksel", "MN60", "Jonah", "Topless", "MN105",
    ],
    3: [  # light grey, n=7
        "Beak", "Fish", "Oscar", "TR77", "SN96", "Bumper", "PL",
    ],
}


def dolphin_truth_by_label():
    truth = {}
    for gid, names in DOLPHIN_GROUPS.items():
        for name in names:
            truth[name] = gid
    return truth


if __name__ == '__main__':
    t = dolphin_truth_by_label()
    print(f"total dolphins assigned: {len(t)} (expect 62)")
    from collections import Counter
    print("group sizes:", Counter(t.values()), "(expect 23,20,12,7)")
