async (page) => {
  const checks = [];
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const check = (name, passed) => {
    if (!passed) throw new Error(name);
    checks.push({name, passed});
  };
  await page.getByRole("button", {name: "文本检测", exact: true}).click();
  const text = page.getByRole("textbox", {name: "待检测文本", exact: true});
  const analyze = page.getByRole("button", {name: "分析文本", exact: true});
  const original = "The team checked the source files before preparing the data. Each record received a stable identifier, and missing values were counted separately.";
  let arrive, release;
  let calls = 0;
  let arrived = new Promise(resolve => arrive = resolve);
  let gate = new Promise(resolve => release = resolve);
  await page.route("**/api/text/detect", async route => {
    calls++; arrive(); await gate;
    await route.fulfill({json: {risk_score: .3, decision: "uncertain", model_version: "browser-fixture"}});
  });
  try {
    await text.fill(original);
    await analyze.click();
    await arrived;
    await text.fill(original + " The author added a new sentence.");
    check("Typing during analysis keeps the submit button disabled", await analyze.isDisabled());
    await analyze.dispatchEvent("click");
    check("Duplicate text submissions are suppressed", calls === 1);
  } finally {release();}
  await page.getByText("输入已更新", {exact: true}).waitFor();
  check("Old text results are discarded and submission recovers", await analyze.isEnabled());
  await page.screenshot({path: "output/playwright/stale-text-prevented.png", fullPage: true});
  await page.unroute("**/api/text/detect");
  await text.fill(original);
  await analyze.click();
  await page.locator("#text-result .result-header").waitFor();
  check("A fresh text analysis renders actual model output", await analyze.isEnabled());
  await page.screenshot({path: "output/playwright/text-analysis.png", fullPage: true});

  await page.getByRole("button", {name: "可信编辑", exact: true}).click();
  const edit = page.locator("#edit-input");
  const rule = page.locator("#run-edit");
  const llm = page.locator("#run-llm-edit");
  calls = 0;
  arrived = new Promise(resolve => arrive = resolve);
  gate = new Promise(resolve => release = resolve);
  await page.route("**/api/text/edit", async route => {
    calls++; arrive(); await gate;
    await route.fulfill({json: {edited_text: original, suggestions: []}});
  });
  try {
    await edit.fill(original);
    await rule.click();
    await arrived;
    await edit.fill("Short");
    check("Both editor actions stay disabled while editing", await rule.isDisabled() && await llm.isDisabled());
    await rule.dispatchEvent("click");
    await llm.dispatchEvent("click");
    check("Rule and LLM editing cannot race", calls === 1);
  } finally {release();}
  await page.getByText("输入已更新", {exact: true}).waitFor();
  check("Editor validates current input when the old request finishes", await rule.isDisabled() && await llm.isDisabled());
  await page.unroute("**/api/text/edit");
  await edit.fill(original);
  check("Valid editor input re-enables both actions", await rule.isEnabled() && await llm.isEnabled());
  await rule.click();
  await page.locator("#edit-output").waitFor();
  check("Actual rule editing renders successfully", (await page.locator("#edit-output").innerText()).length > 20);

  await page.getByRole("button", {name: "图像检测", exact: true}).click();
  await page.getByRole("button", {name: "真实类样本缩略图 真实类样本", exact: true}).click();
  await page.getByRole("button", {name: "分析图片", exact: true}).waitFor();
  const imageButton = page.getByRole("button", {name: "分析图片", exact: true});
  calls = 0;
  arrived = new Promise(resolve => arrive = resolve);
  gate = new Promise(resolve => release = resolve);
  await page.route("**/api/image/detect", async route => {
    calls++; arrive(); await gate;
    await route.fulfill({json: {risk_score: .3, decision: "uncertain", model_version: "browser-fixture"}});
  });
  try {
    await imageButton.click();
    await arrived;
    await page.getByRole("button", {name: "生成类样本缩略图 生成类样本", exact: true}).click();
    await page.getByText("cifake-fake-sample.png", {exact: true}).waitFor();
    check("Changing image cannot re-enable a pending submission", await imageButton.isDisabled());
    await imageButton.dispatchEvent("click");
    check("Duplicate image submissions are suppressed", calls === 1);
  } finally {release();}
  // The route response must finish before the next request uses the new file.
  await page.waitForFunction(() => !document.getElementById("detect-image").disabled);
  check("Old image output does not replace the new image", await page.getByText("图片已就绪", {exact: true}).isVisible());
  await page.unroute("**/api/image/detect");
  await imageButton.click();
  await page.locator("#image-result .result-header").waitFor();
  await page.screenshot({path: "output/playwright/image-analysis.png", fullPage: true});
  check("Actual image model renders for the new image", await imageButton.isEnabled());
  check("No uncaught browser errors", errors.length === 0);
  return {checks, errors};
}
