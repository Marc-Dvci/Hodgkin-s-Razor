"""List every file in a GIN repository folder by walking its HTML tree view."""
import re
import sys
import urllib.request

BASE = "https://gin.g-node.org"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def walk(repo, path):
    html = get(f"{BASE}/{repo}/src/master/{path}")
    prefix = f"/{repo}/src/master/{path}/"
    children = sorted({h for h in re.findall(r'href="([^"?#]+)"', html)
                       if h.startswith(prefix) and "/" not in h[len(prefix):]})
    out = []
    for href in children:
        sub = href.split("/src/master/", 1)[1]
        if re.search(r"\.[A-Za-z0-9]{1,5}$", sub):
            out.append(sub)
        else:
            out += walk(repo, sub)
    return out


if __name__ == "__main__":
    for f in walk(sys.argv[1], sys.argv[2]):
        print(f)
