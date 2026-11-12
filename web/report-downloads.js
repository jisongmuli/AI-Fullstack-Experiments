/* Shared report downloads for the document and Agent views. */
(() => {
  'use strict';
  const pending = new Map();

  async function transfer(reportId, format) {
    const response = await fetch(`/api/document/reports/${encodeURIComponent(reportId)}/${format}`, {
      cache: 'no-store', signal: AbortSignal.timeout(30000)
    });
    if (response.status === 404 || response.status === 410) throw Error('报告已过期或已删除，请重新检测后下载。');
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw Error(typeof error.detail === 'string' ? error.detail : `报告下载失败（${response.status}），请重试。`);
    }
    const blob = await response.blob();
    const encoded = (response.headers.get('Content-Disposition') || '').match(/filename\*=UTF-8''([^;]+)/i);
    let filename = `文档检测报告.${format}`;
    if (encoded) { try { filename = decodeURIComponent(encoded[1]); } catch (_) { /* Keep the fallback filename. */ } }
    const url = URL.createObjectURL(blob), anchor = document.createElement('a');
    anchor.href = url; anchor.download = filename; anchor.hidden = true;
    try { document.body.append(anchor); anchor.click(); }
    finally { anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
    return filename;
  }

  async function download(reportId, format) {
    if (typeof reportId !== 'string' || !reportId || reportId.length > 128 || !['docx', 'html', 'json'].includes(format)) throw Error('报告编号或下载格式无效。');
    const key = `${reportId}:${format}`;
    if (pending.has(key)) return pending.get(key);
    const task = transfer(reportId, format);
    pending.set(key, task);
    try { return await task; }
    finally { if (pending.get(key) === task) pending.delete(key); }
  }

  window.AIGCReports = Object.freeze({download});
})();
