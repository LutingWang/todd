# todd-torch

`HTMLVisual` 导出 PDF/PNG 前，需额外安装 Playwright 的 Chromium 浏览器：

```shell
python -m playwright install chromium
```

调用方负责 Playwright 和 Chromium 的生命周期，并可将同一个 Browser
传给多个导出。`HTMLVisual` 支持直接导出 PNG，输出尺寸与画布一致：

```python
with (
    sync_playwright() as playwright,
    playwright.chromium.launch() as browser,
):
    visual.export_pdf(browser, path=Path('result.pdf'))
    visual.export_screenshot(browser, path=Path('result.png'))
```

需要自定义导出参数或复用同一次渲染时，可通过 `open()` 获取已渲染的 Page；退出上下文时会关闭 Page 并清理临时 HTML：

```python
with visual.open(browser) as page:
    page.pdf(
        path='result.pdf',
        print_background=True,
        prefer_css_page_size=True,
    )
    page.screenshot(path='result.png')
```
