"""
HTML fetcher with trafilatura primary and Playwright fallback.
Playwright is only invoked when trafilatura returns suspiciously little content.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp
import trafilatura

logger = logging.getLogger(__name__)

# Minimum character threshold — below this we suspect JS rendering is required
_JS_THRESHOLD = 500
_FETCH_TIMEOUT_S = 8


@dataclass
class FetchResult:
    url: str
    html: Optional[str]
    used_playwright: bool
    error: Optional[str] = None
    status_code: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.html is not None and len(self.html) >= _JS_THRESHOLD


async def _fetch_with_aiohttp(url: str) -> tuple[Optional[str], Optional[int], Optional[str]]:
    timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT_S)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as resp:
                status = resp.status
                # Consume the body to allow connection reuse even on errors.
                text = await resp.text(errors="replace")
                if status in (403, 404):
                    return None, status, f"HTTP {status}"
                if status >= 400:
                    return None, status, f"HTTP {status}"
                return text, status, None
    except asyncio.TimeoutError:
        return None, None, f"Timeout after {_FETCH_TIMEOUT_S}s"
    except aiohttp.ClientError as e:
        return None, None, f"aiohttp error: {e}"


async def _fetch_with_playwright(url: str) -> Optional[str]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("Playwright not installed — skipping JS fallback")
        return None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.set_extra_http_headers({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            })
            await page.goto(url, wait_until="networkidle", timeout=_FETCH_TIMEOUT_S * 1000)
            html = await page.content()
            return html
        except Exception as e:
            logger.warning("Playwright fetch failed for %s: %s", url, e)
            return None
        finally:
            await browser.close()


async def fetch(url: str) -> FetchResult:
    html, status, err = await _fetch_with_aiohttp(url)

    if err:
        return FetchResult(url=url, html=None, used_playwright=False, error=err, status_code=status)

    if html and len(html) >= _JS_THRESHOLD:
        return FetchResult(url=url, html=html, used_playwright=False, status_code=status)

    logger.debug("trafilatura returned short content for %s — trying Playwright", url)
    html = await _fetch_with_playwright(url)

    if html and len(html) >= _JS_THRESHOLD:
        return FetchResult(url=url, html=html, used_playwright=True, status_code=status)

    return FetchResult(
        url=url,
        html=html,
        used_playwright=True,
        status_code=status,
        error="Content too short or fetch failed",
    )


def fetch_sync(url: str) -> FetchResult:
    return asyncio.run(fetch(url))
