#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
黄鹤一去不复返，白云千载空悠悠。
孤帆远影碧空尽，惟见长江天际流。
"""

import functools
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from datetime import datetime
from multiprocessing import active_children
from pathlib import Path
from typing import (
    Dict,
    Generator,
    Tuple,
    TypeAlias,
    TypeVar,
)

# pip install asynctor requests fake_useragent pyquery beautifulsoup4 loguru
import asynctor
import fake_useragent
import requests
from bs4 import BeautifulSoup
from loguru import logger
from pyquery import PyQuery as pq

# pip install playwright
# playwright install --with-deps chromium --dry-run
# playwright install --with-deps chromium
with suppress(DeprecationWarning):
    from playwright.sync_api import Error, sync_playwright
    from playwright.sync_api import TimeoutError as PlayTimeoutError

BASE_DIR = Path(__file__).parent.resolve()
URI: TypeAlias = str  # e.g.: '11/1101.html'
ADCODE: TypeAlias = str  # e.g.: 110100000000
AreaDict: TypeAlias = Dict[URI, Tuple[ADCODE, str]]
T = TypeVar("T")
INTERVAL = 5  # seconds
INDEX_URL = "https://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2023/index.html"
if os.getenv("PAD_INDEX_URL") == "1":
    INDEX_URL = INDEX_URL.replace("https://", "https://waketzheng.top/")
    INTERVAL = 0
    print(f'Change index URL to be: {INDEX_URL}')
ALTER_URL = "https://www.mca.gov.cn/mzsj/xzqh/2023/202302xzqh.html"
ERR_MSG = "Please enable JavaScript and refresh the page"
ERR_MSG_CN = "请开启JavaScript并刷新该页"
ERR_404 = "404 Not Found"
ACCESS_LIMIT = "访问验证"
RE_TITLE = re.compile(rf"<title>({ERR_404}|{ACCESS_LIMIT})</title>")
HTML_DIR = Path(__file__).parent.resolve() / "htmls"
if not HTML_DIR.exists():
    HTML_DIR.mkdir()
    print(f"Create folder: {HTML_DIR}")


def is_block_res(text: str) -> bool:
    return ERR_MSG in text or ERR_MSG_CN in text or RE_TITLE.search(text) is not None


def cache(func):
    @functools.wraps(func)
    def run(url, *args, **kw) -> str:
        p = HTML_DIR / url.split("://")[-1]
        if p.exists():
            text = p.read_text("utf-8")
            if not is_block_res(text):
                return text
        for _ in range(2):  # try again if blocked
            text = func(url, *args, **kw)
            if not is_block_res(text):
                break
            time.sleep(2)
        else:
            return ""
        p.parent.mkdir(parents=True, exist_ok=True)
        size = p.write_text(text, encoding="utf-8")
        if kw.get("verbose"):
            print(f"Save to {p} with {size=}")
        return text

    return run


@cache
def http_get(url, session=None) -> str:
    if session is None:
        ua = fake_useragent.UserAgent()
        headers = {"User-Agent": ua.chrome}
        r = requests.get(url, headers=headers)
    else:
        r = session.get(url)
    return r.content.decode()


@contextmanager
def start_play():
    with sync_playwright() as p:
        wait_ms = 50
        with p.chromium.launch(headless=False, slow_mo=wait_ms) as browser:
            page = browser.new_page()
            yield page


def walk(root_url: URI = "", province="", verbose=False) -> None:
    if not root_url:
        # root_url="https://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2023/65.html"
        root_url = "https://waketzheng.top/www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2023/11.html"
        province = "北京市"
    elif isinstance(root_url, int) or root_url.isdigit():
        num = int(root_url)
        root_url = f"https://waketzheng.top/www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2023/{num}.html"
        provinces = "北京市 天津市 河北省 山西省 内蒙古自治区 辽宁省 吉林省".split()
        try:
            province = provinces[num - 11]
        except IndexError:
            province = f"code={num}"
    if verbose:
        print(province, "start at:", datetime.now())
        download_recursive(root_url, province, verbose=verbose)
    if verbose:
        print(f"{province} {root_url.split('/')[-1]} finished at {datetime.now()}")


def url_to_file(url: URI) -> Path:
    return HTML_DIR / url.split("://")[-1]


class Player:
    _page = None
    _browser = None
    _playwright = None

    @classmethod
    def init(cls):
        if cls._page is None:
            cls._playwright = p = sync_playwright().start()
            cls._browser = p.chromium.launch(headless=False, slow_mo=50)
            cls._page = cls._browser.new_page()
        return cls._page

    @classmethod
    def teardown(cls):
        if cls._browser is not None:
            cls._browser.close()
        if cls._playwright is not None:
            cls._playwright.close()


def load_or_fetch(url: URI) -> str:
    if (p := url_to_file(url)).exists():
        text = p.read_text("utf-8")
        if not is_block_res(text):
            return text
    short = url.split("/2023/")[-1]
    logger.debug(f"Going to watch {short} @ {os.getpid()=}")
    page = Player.init()
    try:
        # https://playwright.dev/python/docs/api/class-page#page-goto
        response = page.goto(url)
    except PlayTimeoutError:
        return ""
    else:
        if not response.ok:
            return ""
        if (title := page.title()) == ERR_404:
            return ""
        elif title == ACCESS_LIMIT:
            relax = 2 * 60
            logger.info(f"Title of {short} is {ACCESS_LIMIT}, sleep {relax}")
            time.sleep(relax)
            response = page.goto(url)
        else:
            time.sleep(INTERVAL)
    logger.debug(f"{response.status=}")
    try:
        text = response.text()
    except Error as e:
        # 404页面可能会抛如下异常：
        # Response.text: Protocol error (Network.getResponseBody):
        # No resource with given identifier found
        logger.debug(f"Error: {e}")
        return ""
    except Exception as e:
        print(e)
        print(f"{dir(response)}\n{response.status=}\n{url = }")
        raise e
    else:
        logger.debug(f"{len(text) = }")
        soup = BeautifulSoup(text, "html.parser")
        logger.debug(soup.title)
        if soup.title.string in (ERR_404, ACCESS_LIMIT):
            return ""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    logger.debug(f"Save to {p}")
    return text


def parent_path(url: URI) -> URI:
    return url[: url.rindex("/") + 1]


def parse_links(html: str, province=None) -> Generator[Tuple[URI, str], None, None]:
    if html:
        doc = pq(html)
        all_links = doc("a")
        for i in range(len(all_links)):
            a = all_links.eq(i)
            if not (href := a.attr("href")):
                if province is not None:
                    print(province, "Empty href:", a.text())
                continue
            if not href.endswith(".html"):  # 子链接，如：45.html
                # 过滤掉http://www.miibeian.gov.cn/ 京ICP备05034670号
                continue
            yield (href, a.text())  # ('11.html', '北京市')


def build_area_dict(data: AreaDict) -> AreaDict:
    res: AreaDict = {}
    for url, (adcode, name) in data.items():
        if name.isdigit() and not adcode.isdigit():
            adcode, name = name, adcode
        text = load_or_fetch(url)
        if not text:
            continue
        host = parent_path(url)
        for link, label in parse_links(text):
            url = host + link
            res[url] = res.get(url, ()) + (label,)  # type:ignore
    return res


def download_recursive(root_url: URI, province: str, verbose=False) -> None:
    # 省=>市=>县=>镇=>街
    # province -> city -> county -> town -> street
    logger.debug(f"{province=}; {root_url.split('/')[-1]=}; {os.getpid() = }")
    text = load_or_fetch(root_url)
    host = parent_path(root_url)
    assert host.endswith("/")
    # {'11/1101.html': ('110100000000', '市辖区')}
    cities: AreaDict = {}
    for link, label in parse_links(text, province):
        url = host + link
        cities[url] = cities.get(url, ()) + (label,)  # type:ignore
    if verbose:
        print(province, f"{len(cities) = }")
    counties: AreaDict = build_area_dict(cities)
    if verbose:
        print(province, f"{len(counties) = }")
    if not counties:
        return
    towns: AreaDict = build_area_dict(counties)
    if verbose:
        print(province, f"{len(towns) = }")
    if not towns:
        return
    streets: AreaDict = build_area_dict(towns)
    if verbose:
        print(province, f"{len(streets) = }")
    if not streets:
        return
    houses: AreaDict = build_area_dict(streets)
    if verbose:
        print(province, f"{len(houses) = }")
    Player.teardown()


@asynctor.timeit
def main() -> None:
    # 2024.10.10 09:25 执行耗时：main Cost: 1303.2 seconds
    verbose = False
    start = time.time()
    if sys.argv[1:]:
        if (a1 := sys.argv[1]) in ("-h", "--help"):
            print(__doc__)
        elif a1 == "bj":
            verbose = "-v" in sys.argv or "--verbose" in sys.argv
            walk(verbose=verbose)
        elif a1.isdigit():
            count = int(a1)
            with ProcessPoolExecutor() as executor:
                for i in range(11, 11 + count):
                    executor.submit(walk, str(i), verbose=True)
        elif a1.endswith(".html"):
            if (p := Path(a1)).is_file():
                soup = BeautifulSoup(p.read_text("utf-8"), "html.parser")
                print(soup.prettify())
            elif p.stem.isdigit():
                walk(p.stem, verbose=True)
        else:
            if not a1.startswith("-") and not "A" <= a1[0] <= "z":  # 中文
                first_page = http_get(INDEX_URL)
                ph = {p: h for h, p in parse_links(first_page)}
                logger.debug(ph)
                if h := ph.get(a1):
                    walk(Path(h).stem, verbose=True)
                    return
            print(f"Unkown argument: {a1}")
    else:
        first_page = http_get(INDEX_URL)
        host = parent_path(INDEX_URL)
        with ProcessPoolExecutor() as executor:
            future_province = {
                executor.submit(
                    walk, host + href, province_name, verbose=True
                ): province_name
                for href, province_name in parse_links(first_page)
            }
            all_provinces = set(future_province.values())
            total = len(all_provinces)
            future_timeout = 1.5 * 60
            try:
                for index, future in enumerate(
                    as_completed(future_province, timeout=180 * 60), 1
                ):
                    province = future_province[future]
                    print(province, f"waiting for result with {future_timeout=}")
                    try:
                        res = future.result(timeout=future_timeout)
                    except Exception as exc:
                        print("%r generated an exception: %s" % (province, exc))
                        continue
                    past = time.time() - start
                    all_provinces.discard(province)
                    print("%s take %.1f seconds and got %r." % (province, past, res))
                    print(f"Remain: {all_provinces}" if all_provinces else "All done.")
                    ps = active_children()
                    logger.debug(f"Processes: {ps}; {index}/{total}")
            except TimeoutError as e:
                logger.exception(e)
                print(f"{index}/{total}: {all_provinces}")
            logger.debug(f"Tasks completed, to be closing {executor=}")


if __name__ == "__main__":
    main()
