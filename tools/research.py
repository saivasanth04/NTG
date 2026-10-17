"""Read-only public documentation retrieval with pinned TLS destinations."""
import asyncio
import hashlib
import ipaddress
import socket
import ssl
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit, urljoin


class ResearchError(ValueError):
    pass


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class ResearchTools:
    def __init__(self, search_backend=None, allowed_hosts=()):
        self.search_backend = search_backend
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)

    def _url(self, url):
        try:
            p = urlsplit(url)
            if (p.scheme != "https" or p.hostname not in self.allowed_hosts
                    or p.port not in (None, 443) or p.username or p.password
                    or p.query or p.fragment or len(url) > 1000
                    or any(ord(c) <= 32 or ord(c) > 126 for c in url)):
                raise ValueError()
            return p
        except (ValueError, TypeError):
            raise ResearchError("research_url_rejected") from None

    async def search(self, query, limit=3):
        if self.search_backend is None:
            raise ResearchError("research_unavailable")
        if not isinstance(query, str) or not query.strip() or len(query) > 300 or limit != 3:
            raise ResearchError("invalid_research_query")
        try:
            async with asyncio.timeout(10):
                candidates = await self.search_backend(query, limit=limit)
        except TimeoutError:
            raise
        except Exception:
            # Backend-specific transport/status errors must not leak credentials or crash dispatch.
            raise ResearchError("research_unavailable") from None
        if not isinstance(candidates, list) or len(candidates) > 3:
            raise ResearchError("invalid_search_output")
        accepted = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("title"), str):
                continue
            try:
                self._url(candidate.get("url"))
            except ResearchError:
                continue
            accepted.append({"title": candidate["title"][:200], "url": candidate["url"]})
        return accepted

    async def fetch(self, url):
        try:
            return await self._fetch(url)
        except (ResearchError, TimeoutError):
            raise
        except (OSError, ValueError, UnicodeError, asyncio.IncompleteReadError,
                asyncio.LimitOverrunError):
            raise ResearchError("research_fetch_unavailable") from None

    async def _fetch(self, url):
        async with asyncio.timeout(10):
            for redirect in range(3):
                p = self._url(url)
                addresses = await asyncio.get_running_loop().getaddrinfo(
                    p.hostname, 443, type=socket.SOCK_STREAM)
                if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                    raise ResearchError("research_url_rejected")
                # Connect to the checked IP, retaining original hostname certificate checks.
                reader, writer = await asyncio.open_connection(
                    addresses[0][4][0], 443, ssl=ssl.create_default_context(), server_hostname=p.hostname)
                try:
                    request = (f"GET {p.path or '/'} HTTP/1.0\r\nHost: {p.hostname}\r\n"
                               "Accept: text/html, text/plain\r\nAccept-Encoding: identity\r\n"
                               "Connection: close\r\nUser-Agent: NTG-Research/1\r\n\r\n")
                    writer.write(request.encode("ascii"))
                    await writer.drain()
                    header = await reader.readuntil(b"\r\n\r\n")
                    if len(header) > 8192:
                        raise ResearchError("research_fetch_unavailable")
                    lines = header.decode("iso-8859-1").split("\r\n")
                    status = int(lines[0].split()[1])
                    headers = {}
                    for line in lines[1:]:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            headers[key.lower()] = value.strip()
                    if status in (301, 302, 303, 307, 308):
                        if redirect == 2 or "location" not in headers:
                            raise ResearchError("research_fetch_unavailable")
                        url = urljoin(url, headers["location"])
                        continue
                    if (status != 200 or headers.get("content-encoding", "identity") != "identity"
                            or "transfer-encoding" in headers
                            or headers.get("content-type", "").split(";")[0].strip()
                            not in ("text/html", "text/plain")):
                        raise ResearchError("research_fetch_unavailable")
                    body = bytearray()
                    while chunk := await reader.read(4096):
                        body.extend(chunk)
                        if len(body) > 65536:
                            raise ResearchError("research_fetch_unavailable")
                    if "content-length" in headers and int(headers["content-length"]) != len(body):
                        raise ResearchError("research_fetch_unavailable")
                    text = body.decode("utf-8", "replace")
                    if "text/html" in headers["content-type"]:
                        parser = _Text()
                        parser.feed(text)
                        text = " ".join(parser.parts)
                    text = " ".join(text.split())[:6000]
                    if not text:
                        raise ResearchError("research_no_sources")
                    return {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "content_sha256": hashlib.sha256(body).hexdigest(), "text": text}
                finally:
                    writer.close()
                    # Do not leave TLS close-notify waiting beyond cancellation/deadline.
                    transport = getattr(writer, "transport", None)
                    if transport is not None:
                        transport.abort()
            raise ResearchError("research_fetch_unavailable")
