import asyncio
from pathlib import Path
from requests_html import AsyncHTMLSession

# 民政部（6位省市区编码）
MZ_URL = 'http://preview.www.mca.gov.cn/article/sj/xzqh/2020/2020/202101041104.html'

# 统计局（12位编码）
TZ_URL = 'http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020/index.html'

SEP = '___'

session = AsyncHTMLSession()


def url2file(url: str) -> str:
    return url.split('://')[-1].replace('/', SEP)


async def http_get(url: str) -> None:
    return await session.get(url)


def parse_mz(r):
    pass


async def update_adcode():
    print(f'{url2file(MZ_URL) = }')


async def main():
    await update_adcode()


if __name__ == '__main__':
    asyncio.run(main())
