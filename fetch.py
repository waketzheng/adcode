#!/usr/bin/env python
import functools
import json
import pickle
import sys
from pathlib import Path
from typing import TypedDict

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


class ProvinceDict(TypedDict):
    name: str
    url: str
    page: str


doc = pq(r.content.decode())
all_links = doc("a")
provinces: list[ProvinceDict] = []

for i in range(len(all_links)):
    a = all_links.eq(i)
    href = a.attr("href")  # 子链接，如：45.html
    if not (p := Path(href)).stem.isdigit():
        # 过滤掉http://www.miibeian.gov.cn/ 京ICP备05034670号
        continue
    province_name = a.text()
    province_url = url.rsplit("/", 1)[0] + "/" + p.name
    response = http_get(province_url)
    province_page = response.content.decode()
    provinces.append(
        {"name": province_name, "url": province_url, "page": province_page}
    )

if verbose:
    print(f"Text of {province_url}:\n{province_page}")
province_page_pretty = BeautifulSoup(province_page, "html.parser").prettify()
filename = Path(province_url).stem + "_" + province_name + ".html"
size = (p := html_dir / filename).write_text(province_page_pretty)
print(f"Write to {p} with {size=}")
