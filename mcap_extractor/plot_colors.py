import hashlib

FALLBACK_COLORS = [
    "tab:red",
    "tab:orange",
    "tab:blue",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
]


def build_color_map(names):
    """Assign colors deterministically by sorted trajectory names.

    With three names this yields exactly red, green, blue.
    """
    ordered_names = sorted(set(names))
    return {
        name: FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
        for idx, name in enumerate(ordered_names)
    }


def color_for_name(name):
    # Use a deterministic hash so fallback colors stay stable across runs.
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]