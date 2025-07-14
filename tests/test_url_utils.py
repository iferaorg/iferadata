from ifera.enums import Scheme, Source
from ifera import url_utils, file_utils


def test_make_url_file_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils.settings, "DATA_FOLDER", str(tmp_path))
    url = url_utils.make_url(Scheme.FILE, Source.TENSOR, "fut", "1m", "SYM")
    assert url == "file:tensor/fut/1m/SYM.pt.gz"


def test_make_url_tensor_backadjusted(tmp_path, monkeypatch):
    monkeypatch.setattr(file_utils.settings, "DATA_FOLDER", str(tmp_path))
    url = url_utils.make_url(
        Scheme.FILE, Source.TENSOR_BACKADJUSTED, "futures", "1m", "SYM"
    )
    assert url == "file:tensor_backadjusted/futures/1m/SYM.pt.gz"


def test_make_url_other_scheme():
    url = url_utils.make_url(Scheme.S3, Source.RAW, "data", "1d", "ABC")
    assert url == "s3:raw/data/1d/ABC.zip"


def test_extract_date_and_clean_date():
    html = """
    <table>
        <tr><td>First Notice Date</td><td>01/15/2024</td></tr>
        <tr><td>Expiration Date</td><td>02/20/2024</td></tr>
    </table>
    """
    soup = url_utils.BeautifulSoup(html, "html.parser")
    first_notice = url_utils._extract_date("First Notice Date", soup)
    expiration = url_utils._extract_date("Expiration Date", soup)
    assert first_notice is not None and first_notice.isoformat() == "2024-01-15"
    assert expiration is not None and expiration.isoformat() == "2024-02-20"


class DummyResponse:
    def __init__(self, text: str, status_code: int) -> None:
        self.text = text
        self.status_code = status_code


def test_parse_contract_page():
    html = """
    <table>
        <tr><td>First Notice Date</td><td>01/15/2024</td></tr>
        <tr><td>Expiration Date</td><td>02/20/2024</td></tr>
    </table>
    """
    first_notice, expiration = url_utils._parse_contract_page(html)
    assert first_notice is not None and first_notice.isoformat() == "2024-01-15"
    assert expiration is not None and expiration.isoformat() == "2024-02-20"


def test_contract_notice_and_expiry_success(monkeypatch):
    html = """
    <table>
        <tr><td>First Notice Date</td><td>01/15/2024</td></tr>
        <tr><td>Expiration Date</td><td>02/20/2024</td></tr>
    </table>
    """

    def fake_get(*_args, **_kwargs):
        return DummyResponse(html, 200)

    monkeypatch.setattr(
        url_utils, "requests", type("R", (), {"get": staticmethod(fake_get)})()
    )

    first_notice, expiration = url_utils.contract_notice_and_expiry(
        "CLF24", max_retries=1
    )
    assert first_notice is not None and first_notice.isoformat() == "2024-01-15"
    assert expiration is not None and expiration.isoformat() == "2024-02-20"


def test_fetch_contract_page(monkeypatch):
    html = "HTML"

    def fake_get(*_args, **_kwargs):
        return DummyResponse(html, 200)

    monkeypatch.setattr(
        url_utils, "requests", type("R", (), {"get": staticmethod(fake_get)})()
    )

    result = url_utils._fetch_contract_page("CLF24", max_retries=1)
    assert result == html


def test_contract_notice_and_expiry_not_found(monkeypatch):
    def fake_get(*_args, **_kwargs):
        return DummyResponse("", 404)

    monkeypatch.setattr(
        url_utils, "requests", type("R", (), {"get": staticmethod(fake_get)})()
    )

    first_notice, expiration = url_utils.contract_notice_and_expiry(
        "CLF24", max_retries=1
    )
    assert first_notice is None
    assert expiration is None
