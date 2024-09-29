#!/usr/bin/env python
import functools
import json
import pickle
import sys
from pathlib import Path
from typing import Generator, TypedDict

# Python3.10+ is required
# pip install asynctor redis requests fake_useragent pyquery beautifulsoup4
import fake_useragent
import redis
import requests
from bs4 import BeautifulSoup
from pyquery import PyQuery as pq

ONE_YEAR = 365 * 24 * 60 * 60


@functools.lru_cache
def redis_cli():
    return redis.StrictRedis()


def cache(func):
    @functools.wraps(func)
    def run(*args, **kw):
        key = f"{func.__name__}(*{args!r}, **{kw!r})"
        if v := redis_cli().get(key):
            return pickle.loads(v)
        rv = func(*args, **kw)
        redis_cli().set(key, pickle.dumps(rv), ONE_YEAR)
        return rv

    return run


ua = fake_useragent.UserAgent()
session = requests.Session()
session.headers["User-Agent"] = ua.chrome
verbose = "-v" in sys.argv or "--verbose" in sys.argv


@cache
def http_get(url):
    return session.get(url)


url = "https://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2023/index.html"
host = url.rsplit("/", 1)[0]
r = http_get(url)
if verbose:
    print("Response headers:", json.dumps(dict(r.headers)), sep="\n")
html_dir = Path(__file__).parent.resolve() / "htmls"
if not html_dir.exists():
    html_dir.mkdir()
    print(f"Create folder: {html_dir}")
size = (p := html_dir / Path(url).name).write_bytes(r.content)
print(f"Write to {p} with {size=}")
# print(chardet.detect(r.content))
# {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}


class AdcodeDict(TypedDict):
    name: str
    href: str
    adcode: str


class CountyDict(AdcodeDict):
    pass


class CityDict(AdcodeDict):
    # counties: list[CountyDict]
    pass


class ProvinceDict(TypedDict):
    name: str
    url: str
    page: str
    cities: dict


def parse_links(html: str, province=None) -> Generator[tuple[str, str], None, None]:
    doc = pq(html)
    all_links = doc("a")
    for i in range(len(all_links)):
        a = all_links.eq(i)
        if not (href := a.attr("href")):
            print(province, "Empty href:", a.text())
            continue
        if not href.endswith(".html"):  # 子链接，如：45.html
            # 过滤掉http://www.miibeian.gov.cn/ 京ICP备05034670号
            continue
        yield (a.text(), href)


provinces: list[ProvinceDict] = []

for text, href in parse_links(r.content.decode()):
    province_name = text
    province_url = host + "/" + Path(href).name
    response = http_get(province_url)
    province_page = response.content.decode()
    province_index = Path(href).stem  # 65.html -> 65
    cities: dict[str, tuple[str, str]] = {}
    for text, link in parse_links(province_page, text):
        cities[link] = cities.get(link, ()) + (text,)  # type:ignore
    province_host = province_url.rsplit("/", 1)[0]
    for link, (adcode, name) in cities.items():
        if name.isdigit() and not adcode.isdigit():
            adcode, name = name, adcode
        url = province_host + "/" + link
        r = http_get(url)
        r.raise_for_status()
        counties: dict = {}  # 县
        for txt, lnk in parse_links(r.content.decode()):
            counties[lnk] = counties.get(lnk, ()) + (txt,)
        city_host = url.rsplit("/", 1)[0]
        for lnk, (adc, n) in counties.items():
            url = city_host + "/" + lnk
            r = http_get(url)
            r.raise_for_status()
            print(lnk, adc, n)
    provinces.append(
        {
            "name": province_name,
            "url": province_url,
            "page": province_page,
            "cities": cities,
        }
    )

if verbose:
    print(f"Text of {province_url}:\n{province_page}")
province_page_pretty = BeautifulSoup(province_page, "html.parser").prettify()
filename = Path(province_url).stem + "_" + province_name + ".html"
size = (p := html_dir / filename).write_text(province_page_pretty)
print(f"Write to {p} with {size=}")

# 叶子页面（里面没有子链接）
leaf = r.content.decode()
leaf_doc = pq(leaf)
vtr = leaf_doc(".villagetr")
a, b, c = vtr[0].findall("td")
print(a.text, b.text, c.text)
# ('650102002002', '111', '燕儿窝南社区居委会')
