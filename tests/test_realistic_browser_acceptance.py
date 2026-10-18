"""Optional headless Chromium acceptance of the real generated UI.

Only run when SWAAG_BROWSER_ACCEPTANCE=1 and a local browser/driver exists.
This is an offline synthetic UI test: no AI model or external web service.
"""
from __future__ import annotations

import functools
import os
import shutil
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest



@pytest.mark.skipif(os.environ.get("SWAAG_BROWSER_ACCEPTANCE") != "1", reason="optional physical browser acceptance")
def test_operator_portal_real_chromium_interaction(tmp_path: Path):
    selenium = pytest.importorskip("selenium")
    from selenium.webdriver import Chrome, ChromeOptions
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import Select, WebDriverWait

    browser_path = os.getenv("SWAAG_CHROMIUM_BINARY") or shutil.which("chromium-browser") or shutil.which("chromium")
    driver_path = os.getenv("SWAAG_CHROMEDRIVER") or shutil.which("chromedriver")
    if not browser_path or not driver_path:
        pytest.skip("Local Chrome binary and chromedriver not both available")
    explicit_fixture = os.environ.get("SWAAG_REALISTIC_UI_FIXTURE")
    if explicit_fixture:
        # Use a byte-preserved fixture copied from the source host when this
        # machine has Selenium/Chromium but not all SWAAG Python dependencies.
        root = Path(explicit_fixture).resolve() / "work/operator-portal"
    else:
        from swaag.benchmark.realistic_workflows import realistic_benchmark_tasks
        tasks = {task.task_id: task for task in realistic_benchmark_tasks()}
        scenario = tasks["realistic_operator_portal_ui"].create(tmp_path)
        root = scenario.workspace / "work/operator-portal"
    assert (root / "src/status-editor.js").is_file()
    assert (root / "src/bootstrap.js").is_file()
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    options = ChromeOptions()
    options.binary_location = browser_path
    options.add_argument("--headless=new")
    if os.environ.get("SWAAG_CHROME_NO_SANDBOX") == "1":
        # Only for isolated test binaries on machines where Chromium's own
        # sandbox cannot be enabled; never a production browser setting.
        options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1000,800")
    options.add_argument("--no-first-run")
    profile_parent = os.getenv("SWAAG_BROWSER_PROFILE_PARENT") or str(tmp_path)
    driver = None
    try:
        with tempfile.TemporaryDirectory(prefix="swaag-chrome-", dir=profile_parent) as profile:
            options.add_argument("--user-data-dir=" + profile)
            driver = Chrome(service=Service(executable_path=driver_path), options=options)
            # A same-path JavaScript edit must be reloaded from source, not
            # hidden behind Chromium's HTTP/script cache.
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setCacheDisabled", {"cacheDisabled": True})
            target = "http://127.0.0.1:" + str(server.server_port) + "/"

            def run_ui_round(status: str) -> tuple[str, str, str]:
                driver.get(target)
                WebDriverWait(driver, 8).until(lambda b: b.find_element(By.ID, "jobStatus"))
                Select(driver.find_element(By.ID, "jobStatus")).select_by_visible_text(status)
                label = driver.find_element(By.ID, "statusLabel")
                assert label.get_attribute("aria-live") == "polite"
                assert label.text == status
                button = driver.find_element(By.ID, "saveButton")
                assert button.is_displayed() and button.size["height"] >= 35
                button.click()
                saved = driver.find_element(By.ID, "savedStatus")
                return saved.text, driver.execute_script("return localStorage.getItem('operatorStatus')"), label.text

            # The real browser must expose the original defect, not merely run a Node stub.
            original_saved, original_storage, original_label = run_ui_round("running")
            assert original_label == "running"
            assert (original_saved, original_storage) == ("Saved: queued", "queued")

            ui = root / "src/status-editor.js"
            current = ui.read_text(encoding="utf-8")
            old = "    currentLabel.textContent = picker.value;\n"
            assert current.count(old) == 1
            ui.write_text(current.replace(old, old + "    selected = picker.value;\n"), encoding="utf-8")
            for value in ("running", "finished"):
                saved, storage, label = run_ui_round(value)
                assert (saved, storage, label) == ("Saved: " + value, value, value)
            assert driver.find_element(By.ID, "savedStatus").get_attribute("aria-live") == "polite"
    finally:
        if driver is not None:
            driver.quit()
        server.shutdown()
        server.server_close()
        server_thread.join(5)
