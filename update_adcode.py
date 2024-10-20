#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
但使龙城飞将在，不教胡马渡阴山。
多情自古伤离别，可怜花发生。
"""

import functools
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from datetime import datetime
from enum import IntEnum, auto
from multiprocessing import active_children
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Type, TypeAlias, TypeVar, Union

import asyncer
import asynctor
import fake_useragent
import requests

# pip install asynctor requests fake_useragent pyquery beautifulsoup4 loguru tqdm
import tortoise
from bs4 import BeautifulSoup
from database_url import generate
from loguru import logger
from pyquery import PyQuery as pq
from tortoise import Model, Tortoise, fields, run_async
from tortoise.fields.base import StrEnum
from tortoise.fields.data import IntEnumFieldInstance

# pip install playwright
# playwright install --with-deps chromium --dry-run
# playwright install --with-deps chromium
with suppress(DeprecationWarning):
    from playwright.sync_api import Error, sync_playwright
    from playwright.sync_api import TimeoutError as PlayTimeoutError


class AdcodeModel(Model):
    id = fields.IntField(primary_key=True)
    adcode = fields.CharField(max_length=20, unique=True)
    name = fields.CharField(max_length=50)

    class Meta:
        abstract = True


class Province(AdcodeModel):
    name = fields.CharField(max_length=20, unique=True)

    class Meta:
        verbose_name = "省"


class City(AdcodeModel):
    province: fields.ForeignKeyRelation = fields.ForeignKeyField(
        "models.Province", on_delete=fields.OnDelete.CASCADE, related_name="cities"
    )

    class Meta:
        verbose_name = "市"


class County(AdcodeModel):
    city: fields.ForeignKeyRelation = fields.ForeignKeyField(
        "models.City", on_delete=fields.OnDelete.CASCADE, related_name="counties"
    )

    class Meta:
        verbose_name = "县"


class Town(AdcodeModel):
    county: fields.ForeignKeyRelation = fields.ForeignKeyField(
        "models.County", on_delete=fields.OnDelete.CASCADE, related_name="towns"
    )

    class Meta:
        verbose_name = "镇"


class Village(AdcodeModel):
    town: fields.ForeignKeyRelation = fields.ForeignKeyField(
        "models.Town", on_delete=fields.OnDelete.CASCADE, related_name="villages"
    )
    ur_code = fields.CharField(3, description="3位城乡属性划分代码", default="")

    class Meta:
        verbose_name = "乡"


"""
| Column       | Type                  | Description                                             |
| ------------ | --------------------- | ------------------------------------------------------- |
| code         | bigint                | 国家统计局12位行政区划代码                              |
| parent       | bigint                | 12位父级行政区划代码                                    |
| name         | character varying(64) | 行政单位名称                                            |
| level        | character varying(16) | 行政单位级别:国/省/市/县/乡/村                          |
| rank         | integer               | 行政单位级别{0:国,1:省,2:市,3:区/县,4:乡/镇，5:街道/村} |
| adcode       | integer               | 6位县级行政区划代码                                     |
| post_code    | character varying(8)  | 邮政编码                                                |
| area_code    | character varying(4)  | 长途区号                                                |
| ur_code      | character varying(4)  | 3位城乡属性划分代码                                     |
| municipality | boolean               | 是否为直辖行政单位                                      |
| virtual      | boolean               | 是否为虚拟行政单位，例如市辖区、省直辖县等。            |
| dummy        | boolean               | 是否为模拟行政单位，例如虚拟社区、虚拟村。              |
| longitude    | double precision      | 地理中心经度                                            |
| latitude     | double precision      | 地理中心纬度                                            |
| center       | geometry              | 地理中心, `ST_Point`                                    |
| province     | character varying(64) | 省                                                      |
| city         | character varying(64) | 市                                                      |
| county       | character varying(64) | 区/县                                                   |
| town         | character varying(64) | 乡/镇                                                   |
| village      | character varying(64) | 街道/村                                                 |
"""


class AutoName(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class LevelEnum(AutoName):
    """Use name as value

    Usage::
        >>> LevelEnum.country == 'country'
        True
        >>> LevelEnum.city.name == LevelEnum.city.value == LevelEnum.city == 'city'
        True
    """

    country = auto()
    province = auto()
    city = auto()
    county = auto()
    town = auto()
    village = auto()


class RankEnum(IntEnum):
    country = 0
    province = 1
    city = 2
    county = 3
    town = 4
    village = 5


class BoolNullEnum(StrEnum):
    true = "t"
    false = "f"
    null = ""


def BoolField(verbose_name: str, **kwargs) -> BoolNullEnum:
    kwargs.setdefault("default", BoolNullEnum.false)
    return fields.CharEnumField(BoolNullEnum, verbose_name=verbose_name, **kwargs)


class RankFieldInstance(IntEnumFieldInstance):
    def to_python_value(self, value: Union[str, int, None]) -> Union[IntEnum, None]:
        if isinstance(value, str):
            value = int(value)
        return super().to_python_value(value)


def RankField(verbose_name: str, **kwargs) -> RankFieldInstance:
    return RankFieldInstance(RankEnum, verbose_name, **kwargs)


class AreaInfo(Model):
    id = fields.IntField(primary_key=True)
    code = fields.IntField(verbose_name="国家统计局12位行政区划代码")
    parent = fields.IntField(verbose_name="12位父级行政区划代码")
    name = fields.CharField(max_length=64, verbose_name="行政单位名称", default="")
    level = fields.CharEnumField(LevelEnum, verbose_name="行政单位级别(英文)")
    rank = RankField("行政单位级别(数值)")
    adcode = fields.IntField(verbose_name="6位县级行政区划代码")
    post_code = fields.CharField(max_length=8, verbose_name="邮政编码")
    area_code = fields.CharField(max_length=4, verbose_name="长途区号")
    ur_code = fields.CharField(max_length=4, verbose_name="3位城乡属性划分代码")
    municipality = BoolField("是否为直辖行政单位")
    virtual = BoolField("是否为虚拟行政单位，例如市辖区、省直辖县等。")
    dummy = BoolField("是否为模拟行政单位，例如虚拟社区、虚拟村。")
    longitude = fields.CharField(max_length=32, verbose_name="地理中心经度", default="")
    latitude = fields.CharField(max_length=32, verbose_name="地理中心纬度", default="")
    center = fields.TextField(verbose_name="地理中心, `ST_Point`", default="")
    province = fields.CharField(max_length=64, verbose_name="省", default="")
    city = fields.CharField(max_length=64, verbose_name="市", default="")
    county = fields.CharField(max_length=64, verbose_name="区/县", default="")
    town = fields.CharField(max_length=64, verbse_name="乡/镇", default="")
    village = fields.CharField(max_length=64, verbose_name="街道/村", default="")

    @classmethod
    def sorted_fields(cls) -> list[str]:
        fields = """
        code
        parent
        name
        level
        rank
        adcode
        post_code
        area_code
        ur_code
        municipality
        virtual
        dummy
        longitude
        latitude
        center
        province
        city
        county
        town
        village
        """
        return [j for i in fields.strip().splitlines() if (j := i.strip()) != "id"]


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
    print(f"Change index URL to be: {INDEX_URL}")
# 注：ALTER_URL用不上，因为INDEX_URL获取到的已经是变更后的了
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
    _page = None  # type:ignore
    _browser = None  # type:ignore
    _playwright = None  # type:ignore

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


async def parse_area_dict(
    data: AreaDict,
    model: Type[AdcodeModel],
    parent_model: Type[AdcodeModel],
    css_class: str,
    adcode_obj: Optional[dict[str, AdcodeModel]] = None,
) -> AreaDict:
    res: AreaDict = {}
    if adcode_obj is None:
        objs = await parent_model.all()
        adcode_obj = {i.adcode: i for i in objs}
    attr = parent_model.__name__.lower()
    for url, (adcode, name) in data.items():
        if name.isdigit() and not adcode.isdigit():
            adcode, name = name, adcode
        text = load_or_fetch(url)
        if not text:
            continue
        try:
            parent = adcode_obj[adcode]
        except KeyError:
            if obj := await parent_model.filter(adcode=adcode).first():
                parent = obj
            else:
                print(f"Object not found: {parent_model.__name__}({adcode = })")
                continue
        await create_objects(text, parent, attr=attr, css_class=css_class, model=model)
        host = parent_path(url)
        for link, label in parse_links(text):
            url = host + link
            res[url] = res.get(url, ()) + (label,)  # type:ignore
    return res


async def create_objects(
    text: str,
    parent: AdcodeModel,
    attr: str = "province",
    css_class: str = "tr.citytr",
    model: Type[AdcodeModel] = City,
) -> dict:
    adcode_obj: dict = {}
    doc = pq(text)
    if not (trs := list(doc(css_class).items())) and css_class == "tr.countytr":
        trs = list(doc("tr.towntr").items())
    for tr in trs:
        parts = tr.text().split()
        try:
            adcode, *ur_code, name = parts
        except ValueError:
            print(f"Can't parse adcode: {parts}")
            continue
        if (obj := await model.filter(adcode=adcode).first()) is None:
            obj = model(name=name, adcode=adcode)
            setattr(obj, attr, parent)
            if ur_code and model is Village:
                obj.ur_code = ur_code[0]
            try:
                await obj.save()
            except tortoise.exceptions.IntegrityError as e:
                print(f"{attr=}; {parent=}; {obj=}")
                raise e
        adcode_obj[adcode] = obj
    return adcode_obj


async def parse_downloaded(root_url: URI, province: str, verbose=False) -> None:
    logger.debug(f"{province=}; {root_url.split('/')[-1]=}; {os.getpid() = }")
    text = load_or_fetch(root_url)
    host = parent_path(root_url)
    assert host.endswith("/")
    obj, created = await Province.get_or_create(
        name=province, adcode=Path(root_url).stem
    )
    if verbose:
        print(created, obj)
    cities: AreaDict = {}
    for link, label in parse_links(text, province):
        url = host + link
        cities[url] = cities.get(url, ()) + (label,)  # type:ignore
    adcode_city = await create_objects(text, obj)
    if verbose:
        print(province, f"{len(adcode_city) = }")
    counties: AreaDict = await parse_area_dict(
        cities, County, City, "tr.countytr", adcode_city
    )
    if verbose:
        print(province, f"has link: {len(counties) = }")
    if not counties:
        return
    towns: AreaDict = await parse_area_dict(counties, Town, County, "tr.towntr")
    if verbose:
        print(province, f"objects with link: {len(towns) = }")
    if not towns:
        return
    villages: AreaDict = await parse_area_dict(towns, Village, Town, "tr.villagetr")
    if verbose and villages:  # Expected villages to be empty, as them are no link
        print(province, f"{len(villages) = }")
    if not villages:
        return
    houses: AreaDict = build_area_dict(villages)
    if verbose:
        logger.warning(f"{province} -- {len(houses) = }")


def download_recursive(root_url: URI, province: str, verbose=False) -> None:
    # 省=>市=>县=>镇=>乡
    # province -> city -> county -> town -> village
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
    villages: AreaDict = build_area_dict(towns)
    if verbose:
        print(province, f"{len(villages) = }")
    if not villages:
        return
    houses: AreaDict = build_area_dict(villages)
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
        elif a1 == "parse":
            run_async(laving())
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


async def register_orm():
    await Tortoise.init(
        db_url=generate("db.sqlite3"),
        modules={"models": ["__main__"]},
    )
    await Tortoise.generate_schemas()


async def laving() -> None:
    await register_orm()
    first_page = http_get(INDEX_URL)
    host = parent_path(INDEX_URL)
    async with asyncer.create_task_group() as tg:
        for href, province_name in parse_links(first_page):
            tg.soonify(parse_downloaded)(host + href, province_name, verbose=True)
    await read_local_adcodes(verbose=True)
    print("provinces:", await Province.all())
    print("cities:", await City.all().count())
    print("counties:", await County.all().count())
    print("towns:", await Town.all().count())
    print("villages:", await Village.all().count())
    print("area infos:", await AreaInfo.all().count())


def parse_csv(p: Path) -> list[list[str]]:
    text = p.read_text("utf-8")
    lines = text.splitlines()
    return [j.split(",") for i in lines if (j := i.strip())]


async def read_local_adcodes(verbose=False) -> list[AreaInfo]:
    if verbose:
        from tqdm import tqdm as _tqdm

        def tqdm(g):
            return _tqdm(list(g))
    else:

        def tqdm(g):
            return g

    dirpath = BASE_DIR / "data" / "adcode"
    fields = AreaInfo.sorted_fields()
    objs: list[AreaInfo] = []
    for p in tqdm(dirpath.glob("*.csv")):
        for row in parse_csv(p):
            objs.append(AreaInfo(**dict(zip(fields, row))))
    await AreaInfo.bulk_create(objs)
    return objs


if __name__ == "__main__":
    main()
