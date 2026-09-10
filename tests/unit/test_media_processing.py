from __future__ import annotations

import base64
from io import BytesIO

import httpx2
import pytest
from PIL import Image

from vv_llm.chat_clients.utils import format_image_message, format_messages
from vv_llm.types.enums import BackendType
from vv_llm.utilities.media_processing import ImageProcessor


def _write_image(path, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, color=(35, 96, 120))
    image.save(path, format="PNG")
    image.close()


def _image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes))


def _data_url(size: tuple[int, int]) -> str:
    image = Image.new("RGB", size, color=(35, 96, 120))
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image.close()
    return f"data:image/png;base64,{base64.b64encode(image_bytes.getvalue()).decode()}"


@pytest.mark.parametrize("status_code", [301, 302])
def test_image_processor_follows_redirects(monkeypatch, status_code) -> None:
    image_bytes = base64.b64decode(_data_url((4, 2)).split(",", 1)[1])
    paths = []

    def handle_request(request):
        paths.append(request.url.path)
        if request.url.path == "/redirect":
            return httpx2.Response(status_code, headers={"Location": "/image.png"})
        assert request.url.path == "/image.png"
        return httpx2.Response(200, content=image_bytes)

    with httpx2.Client(transport=httpx2.MockTransport(handle_request)) as client:
        monkeypatch.setattr("vv_llm.utilities.media_processing.httpx2.get", client.get)
        processor = ImageProcessor("https://example.com/redirect", max_size=None)
        image = _image_from_bytes(processor.bytes)

    assert paths == ["/redirect", "/image.png"]
    assert image.size == (4, 2)
    assert image.getpixel((0, 0)) == (35, 96, 120)


@pytest.mark.parametrize("status_code", [403, 404])
def test_image_processor_preserves_http_errors(monkeypatch, status_code) -> None:
    image_url = "https://example.com/image.png"
    response = httpx2.Response(status_code, text="Image download failed")

    with httpx2.Client(transport=httpx2.MockTransport(lambda request: response)) as client:
        monkeypatch.setattr("vv_llm.utilities.media_processing.httpx2.get", client.get)
        with pytest.raises(httpx2.HTTPStatusError) as error:
            ImageProcessor(image_url)

    assert error.value.response is response
    assert error.value.response.status_code == status_code
    assert error.value.response.text == "Image download failed"
    assert str(error.value.request.url) == image_url


def test_image_processor_resizes_long_side_without_changing_ratio(tmp_path) -> None:
    image_path = tmp_path / "wide.png"
    _write_image(image_path, (10_000, 500))

    processor = ImageProcessor(image_path, max_size=None, max_image_dimension=8192)
    resized = _image_from_bytes(processor.bytes)

    assert resized.size == (8192, 409)
    assert max(resized.size) <= 8192
    assert processor.bytes is processor.bytes
    assert processor.base64_image == processor.base64_image


def test_image_processor_does_not_apply_file_size_loop_when_unlimited(tmp_path) -> None:
    image_path = tmp_path / "small.png"
    _write_image(image_path, (400, 200))

    processor = ImageProcessor(image_path, max_size=None, max_image_dimension=8192)
    resized = _image_from_bytes(processor.bytes)

    assert resized.size == (400, 200)


def test_image_processor_handles_rgba_images_with_a_dimension_limit() -> None:
    image = Image.new("RGBA", (400, 200), color=(35, 96, 120, 128))
    processor = ImageProcessor(image, max_size=None, max_image_dimension=128)

    resized = _image_from_bytes(processor.bytes)

    assert resized.size == (128, 64)
    assert processor.mime_type == "image/png"


def test_format_image_message_applies_model_dimension_limit(tmp_path) -> None:
    image_path = tmp_path / "message.png"
    _write_image(image_path, (400, 200))

    formatted = format_image_message(str(image_path), max_image_dimension=128)
    encoded_image = formatted["image_url"]["url"].split(",", 1)[1]
    resized = _image_from_bytes(base64.b64decode(encoded_image))

    assert resized.size == (128, 64)


def test_format_messages_resizes_existing_openai_image_url() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": _data_url((400, 200))}},
            ],
        }
    ]

    formatted = format_messages(
        messages,
        backend=BackendType.OpenAI,
        native_multimodal=True,
        function_call_available=True,
        max_image_dimension=128,
    )
    encoded_image = formatted[0]["content"][1]["image_url"]["url"].split(",", 1)[1]
    resized = _image_from_bytes(base64.b64decode(encoded_image))

    assert resized.size == (128, 64)
