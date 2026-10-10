# -*- coding: utf-8 -*-
"""第 10 輪(GitHub Pages + 問句化 + 國際化 + 稽核紀錄)回歸測試。"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

PAGES = ["index.md", "faq.md", "anti-scam.md", "methodology.md",
         "quickstart.md", "audit-trail.md", "user-guide.md"]


def _body_h1_count(text: str) -> int:
    """數「非程式碼區塊內」的 H1(SEO:每頁單一 H1)。"""
    n = 0
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("# "):
            n += 1
    return n


def test_pages_config_valid():
    """docs/_config.yml 必須含正確的 url/baseurl/lang 與 SEO 外掛。"""
    s = io.open(DOCS / "_config.yml", encoding="utf-8").read()
    assert 'url: "https://mars-tw.github.io"' in s
    assert 'baseurl: "/anti-gambling-trader-tw"' in s
    assert "lang: zh-Hant-TW" in s
    assert "jekyll-sitemap" in s and "jekyll-seo-tag" in s


def test_docs_pages_have_front_matter_and_single_h1():
    """每頁要有 front matter(title/description)且正文恰一個 H1。"""
    for name in PAGES:
        s = io.open(DOCS / name, encoding="utf-8").read()
        assert s.startswith("---\n"), f"{name} 缺 front matter"
        fm = s.split("---", 2)[1]
        assert "title:" in fm and "description:" in fm, f"{name} front matter 不完整"
        assert _body_h1_count(s) == 1, f"{name} 的正文 H1 數不是 1"


def test_docs_relative_links_resolve():
    """docs 內的相對 .md 連結必須存在(Pages 與 github.com 都不能有死鏈)。"""
    broken = []
    for f in DOCS.glob("*.md"):
        s = io.open(f, encoding="utf-8").read()
        for m in re.finditer(r"\]\(([^)#]+?\.md)(#[^)]*)?\)", s):
            target = m.group(1)
            if target.startswith("http"):
                continue
            if not (DOCS / target).exists() and not (ROOT / target.lstrip("./")).exists():
                broken.append(f"{f.name} → {target}")
    assert not broken, "死鏈:" + "; ".join(broken)


def test_audit_trail_is_honest_ledger():
    """稽核紀錄必須揭露「非第三方稽核」並含可核對的 commit hash。"""
    s = io.open(DOCS / "audit-trail.md", encoding="utf-8").read()
    assert ("不是第三方" in s or "非第三方" in s), "缺『非第三方稽核』揭露"
    assert re.search(r"[0-9a-f]{7}", s), "缺可核對的 commit hash"
    assert "已知限制" in s, "缺已知限制節"
    assert "回歸" in s, "必須收錄『複核抓到修復回歸』的負面證據"


def test_english_readme_exists_and_faithful_anchors():
    """英文版存在、有語言切換連結、保留誠實聲明。"""
    s = io.open(ROOT / "README.en.md", encoding="utf-8").read()
    assert "README.md" in s  # 語言切換
    assert "165" in s
    assert "pseudonym" in s.lower() or "not a registered" in s.lower(), \
        "英文版必須保留『社群化名/非立案法人』誠實聲明"


def test_docs_pages_have_byline_footer():
    """內容頁要有頁面級署名(維護者/AI 輔助/審閱日期)—— E-E-A-T 要求。"""
    for name in (
        "faq.md", "anti-scam.md", "methodology.md", "audit-trail.md", "user-guide.md"
    ):
        s = io.open(DOCS / name, encoding="utf-8").read()
        assert "維護者:好棒棒反詐協會" in s, f"{name} 缺署名頁尾"
        assert "AI 輔助" in s, f"{name} 缺 AI 使用揭露"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  OK {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)}")
    sys.exit(1 if failed else 0)
