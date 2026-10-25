/* Document analysis uses the server's existing paragraph detectors. No source
 * text is persisted in localStorage. All ranges refer to Unicode codepoints. */
(() => {
  'use strict';

  const MAX_BYTES = 20 * 1024 * 1024;
  const MAX_CHARACTERS = 100000;
  const HISTORY_KEY = 'aigc.document.history.v1';
  const HISTORY_LIMIT = 20;
  const acceptedExtensions = new Set(['docx', 'pdf', 'txt', 'md']);
  const byId = (id) => document.getElementById(id);
  const finite = (value) => typeof value === 'number' && Number.isFinite(value);
  const boundedPercent = (value) => finite(value) ? Math.max(0, Math.min(100, value)) : null;
  const percentText = (value) => boundedPercent(value) === null ? '—' : `${boundedPercent(value).toFixed(1)}%`;
  const icons = () => { if (window.lucide && typeof window.lucide.createIcons === 'function') window.lucide.createIcons(); };
  const make = (tag, className, text) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = String(text);
    return element;
  };
  const codepointCount = (text) => Array.from(text).length;
  const dateText = (value) => {
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? date.toLocaleString('zh-CN', {hour12: false}) : '时间未提供';
  };
  const evidenceText = (value) => {
    if (typeof value === 'string') return value;
    if (value && typeof value === 'object' && typeof value.message === 'string') return value.message;
    return '';
  };

  // Shared with the existing text and Agent pages. Return DOM rather than an
  // interpolated HTML string so document text can never become executable markup.
  function renderInline(text, suppliedSegments, options = {}) {
    const original = typeof text === 'string' ? text : '';
    const points = Array.from(original);
    const wrapper = make('div', 'document-inline-preview');
    const heading = make('div', 'document-inline-heading', '原文下划线标注');
    const reader = make('div', `document-reader${options.showUnscored ? ' show-unscored' : ''}`);
    reader.setAttribute('aria-label', '原文与疑似 AI 特征段落下划线');
    const segments = (Array.isArray(suppliedSegments) ? suppliedSegments : [])
      .filter((item) => item && Number.isInteger(item.start) && Number.isInteger(item.end) && item.start >= 0 && item.end > item.start && item.end <= points.length)
      .sort((a, b) => a.start - b.start);
    let cursor = 0;
    segments.forEach((segment) => {
      if (segment.start < cursor) return;
      if (segment.start > cursor) reader.append(document.createTextNode(points.slice(cursor, segment.start).join('')));
      const span = make('span', `document-span${segment.highlighted === true ? ' document-risk-span' : ''}${!finite(segment.risk_score) ? ' document-unscored-span' : ''}`, points.slice(segment.start, segment.end).join(''));
      const evidence = Array.isArray(segment.evidence) ? segment.evidence.map(evidenceText).filter(Boolean).slice(0, 4) : [];
      const label = finite(segment.risk_score) ? `模型分数 ${percentText(segment.risk_score * 100)}${segment.highlighted === true ? '；高风险段落' : '；未达到高风险阈值'}` : '未评分段落';
      span.title = `${label}${evidence.length ? `\n${evidence.join('\n')}` : ''}`;
      span.setAttribute('aria-label', label);
      if (segment.highlighted === true || !finite(segment.risk_score)) span.tabIndex = 0;
      reader.append(span); cursor = segment.end;
    });
    if (cursor < points.length) reader.append(document.createTextNode(points.slice(cursor).join('')));
    wrapper.append(heading, reader, make('p', 'document-metric-note', '下划线表示整个段落的模型信号，不能确认每个词的真实来源。'));
    return wrapper;
  }
  window.AIGCDocument = Object.freeze({renderInline});

  document.addEventListener('DOMContentLoaded', () => {
    if (!byId('view-document')) return;
    const ui = {
      modeFile: byId('document-mode-file'), modeText: byId('document-mode-text'),
      filePane: byId('document-file-pane'), textPane: byId('document-text-pane'),
      file: byId('document-file'), dropzone: byId('document-dropzone'), fileMeta: byId('document-file-meta'),
      name: byId('document-file-name'), size: byId('document-file-size'), clearFile: byId('document-clear-file'),
      text: byId('document-text'), count: byId('document-character-count'),
      detect: byId('document-detect'), cancel: byId('document-cancel'), status: byId('document-status'),
      result: byId('document-result'), saveHistory: byId('document-save-history'),
      history: byId('document-history'), clearHistory: byId('document-clear-history'),
      reports: byId('document-server-reports'), refreshReports: byId('document-refresh-reports'),
      reportsStatus: byId('document-reports-status'), reportsCount: byId('document-reports-count'),
      jobs: byId('document-jobs'), jobsStatus: byId('document-jobs-status'),
      refreshJobs: byId('document-refresh-jobs'), background: byId('document-background')
    };
    let mode = 'file';
    let selectedFile = null;
    let busy = false;
    let busyAction = '';
    let currentReport = null;
    let controller = null;
    let requestVersion = 0;
    let history = readHistory();
    let serverReports = [];
    let listVersion = 0;
    let listLoading = false;
    let listTimer = null;
    let jobList = [], jobsVersion = 0, jobsTimer = null;
    let activeJobId = null, cancelRequested = false, backgrounded = false;

    function status(message = '', isError = false) {
      ui.status.classList.toggle('error', isError);
      ui.status.textContent = message;
    }

    const terminalJob = job => ['completed', 'failed', 'cancelled'].includes(job.status);
    const jobText = job => ({pending: '等待开始', extracting: '正在提取文字', analyzing: '正在分析',
      saving: '正在生成报告', completed: '检测完成', failed: '处理失败', cancelled: '已取消'}[job.stage] || '正在处理');
    function jobLabel(job) {
      return job.status === 'cancel_requested' ? '正在取消，请稍候…' : `${jobText(job)}${job.stage === 'analyzing' && job.total ? ` · ${job.completed} / ${job.total}` : ''}`;
    }
    function rememberJob(job) {
      jobList = [job, ...jobList.filter(item => item.job_id !== job.job_id)];
      renderJobs();
    }
    function renderJobs() {
      ui.jobs.replaceChildren();
      if (!jobList.length) ui.jobs.append(make('p', 'muted', '暂无文档任务。检测后可在这里查看进度或恢复任务。'));
      const active = jobList.filter(job => !terminalJob(job));
      const visible = [...active, ...jobList.filter(terminalJob).slice(0, Math.max(0, 8 - active.length))];
      visible.forEach(job => {
        const row = make('div', 'document-job-row');
        row.append(make('strong', '', job.filename), make('span', '', jobLabel(job)));
        if (job.status === 'failed') row.append(make('span', 'document-job-error', job.error?.message || '请重新检测。'));
        if (!terminalJob(job)) {
          const bar = make('progress'); bar.max = 100; bar.value = job.progress;
          bar.setAttribute('aria-label', `${job.filename}的处理进度`); row.append(bar);
        }
        const actions = make('div', 'document-job-actions');
        if (!terminalJob(job) || (job.status === 'completed' && job.report_id)) {
          const open = make('button', 'document-link-button', terminalJob(job) ? '查看报告' : '查看进度');
          open.type = 'button'; open.disabled = busy;
          open.setAttribute('aria-label', `${terminalJob(job) ? '查看任务报告' : '恢复任务'}：${job.filename}`);
          open.onclick = () => terminalJob(job) ? restoreReport(job.report_id) : resumeJob(job);
          actions.append(open);
        }
        if (!terminalJob(job)) {
          const cancel = make('button', 'document-link-button', '取消任务');
          cancel.type = 'button'; cancel.disabled = busy || job.status === 'cancel_requested';
          cancel.dataset.cancelling = String(job.status === 'cancel_requested');
          cancel.setAttribute('aria-label', `取消任务：${job.filename}`);
          cancel.onclick = () => requestCancel(job.job_id); actions.append(cancel);
        }
        row.append(actions); ui.jobs.append(row);
      });
    }
    async function refreshJobs() {
      const version = ++jobsVersion;
      clearTimeout(jobsTimer);
      ui.refreshJobs.disabled = true;
      try {
        const response = await fetch('/api/document/jobs', {cache: 'no-store', signal: AbortSignal.timeout(5000)});
        if (!response.ok) throw Error('任务列表读取失败，请刷新重试。');
        const body = await response.json();
        if (!Array.isArray(body.jobs)) throw Error('任务列表格式异常，请刷新重试。');
        if (version !== jobsVersion) return;
        jobList = body.jobs.filter(job => job && typeof job.job_id === 'string' && typeof job.filename === 'string').slice(0, 20);
        ui.jobsStatus.textContent = ''; renderJobs();
      } catch (error) {
        if (version === jobsVersion) ui.jobsStatus.textContent = '任务列表暂时不可用，可点击刷新重试。';
      } finally {
        if (version === jobsVersion) {
          ui.refreshJobs.disabled = false;
          if (!document.hidden && byId('view-document').classList.contains('active')) jobsTimer = setTimeout(refreshJobs, jobList.some(job => !terminalJob(job)) ? 1500 : 15000);
        }
      }
    }
    async function requestCancel(jobId) {
      const known = jobList.find(job => job.job_id === jobId);
      if (known && !terminalJob(known)) rememberJob({...known, status: 'cancel_requested'});
      if (jobId === activeJobId) { cancelRequested = true; updateControls(); }
      try {
        const response = await fetch(`/api/document/jobs/${encodeURIComponent(jobId)}/cancel`, {method: 'POST', signal: AbortSignal.timeout(5000)});
        if (!response.ok) throw Error('任务取消失败，请刷新任务后重试。');
        const job = await response.json(); rememberJob(job);
        if (jobId === activeJobId) {
          cancelRequested = job.status === 'cancel_requested';
          status(job.status === 'completed' ? '任务已完成，可查看报告。' : jobLabel(job)); updateControls();
        }
        refreshJobs();
      } catch (error) {
        if (jobId === activeJobId) { cancelRequested = false; updateControls(); }
        status(error instanceof TypeError ? '取消请求未连接到服务，请重试。' : error.message, true);
        refreshJobs();
      }
    }
    function pause(milliseconds, signal) {
      return new Promise((resolve, reject) => {
        if (signal.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
        const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(new DOMException('Aborted', 'AbortError')); };
        const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, milliseconds);
        signal.addEventListener('abort', abort, {once: true});
      });
    }
    function showJobProgress(job) {
      const bar = byId('document-task-progress'), label = byId('document-task-stage');
      if (bar) { bar.value = job.progress; bar.setAttribute('aria-valuetext', jobLabel(job)); }
      if (label) label.textContent = jobLabel(job);
    }
    async function followJob(job, signal) {
      activeJobId = job.job_id;
      updateControls();
      while (true) {
        if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
        cancelRequested = job.status === 'cancel_requested'; updateControls();
        rememberJob(job); showJobProgress(job);
        if (job.status === 'completed') {
          const response = await fetch(`/api/document/reports/${encodeURIComponent(job.report_id)}`, {signal});
          return responseJson(response);
        }
        if (job.status === 'failed') throw Error(job.error?.message || '文档处理失败，请重新检测。');
        if (job.status === 'cancelled') { const error = Error('任务已取消。'); error.name = 'JobCancelled'; throw error; }
        await pause(750, signal);
        const response = await fetch(`/api/document/jobs/${encodeURIComponent(job.job_id)}`, {signal: AbortSignal.any([signal, AbortSignal.timeout(5000)]), cache: 'no-store'});
        if (!response.ok) throw Error('任务记录暂时无法读取，可从文档任务列表恢复。');
        job = await response.json();
      }
    }

    function readHistory() {
      try {
        const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
        if (!Array.isArray(parsed)) return [];
        return parsed.filter((item) => item && typeof item.report_id === 'string' && item.report_id.length < 160 &&
          typeof item.filename === 'string' && item.filename.length < 1024 && typeof item.created_at === 'string')
          .slice(0, HISTORY_LIMIT).map((item) => ({
            report_id: item.report_id, filename: item.filename, created_at: item.created_at,
            format: typeof item.format === 'string' ? item.format.slice(0, 20) : '',
            highlighted_total_percent: boundedPercent(item.highlighted_total_percent),
            expires_at: typeof item.expires_at === 'string' ? item.expires_at : ''
          }));
      } catch (_) { return []; }
    }

    function saveHistory(report) {
      if (!ui.saveHistory.checked) return;
      const item = {
        report_id: report.report_id,
        filename: report.filename || '粘贴的长文',
        format: report.format || 'text',
        created_at: report.created_at || new Date().toISOString(),
        highlighted_total_percent: boundedPercent(report.document_summary?.highlighted_total_percent),
        expires_at: typeof report.expires_at === 'string' ? report.expires_at : ''
      };
      const next = [item, ...history.filter((entry) => entry.report_id !== item.report_id)].slice(0, HISTORY_LIMIT);
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
        history = next;
        renderHistory();
      } catch (_) {
        ui.saveHistory.checked = false;
        status('浏览器未允许保存记录，本次结果仍可查看和下载。', true);
      }
    }

    function renderHistory() {
      ui.history.replaceChildren();
      if (!history.length) {
        ui.history.append(make('p', 'muted', '尚无本地记录。勾选后才会保存结果摘要。'));
      }
      history.forEach((entry) => {
        const expired = entry.expires_at && new Date(entry.expires_at).getTime() <= Date.now();
        const button = make('button', `document-history-item${expired ? ' expired' : ''}`);
        button.type = 'button';
        button.append(make('strong', '', entry.filename));
        button.append(make('span', '', `${dateText(entry.created_at)} · 特征覆盖率 ${percentText(entry.highlighted_total_percent)}${expired ? ' · 报告已过期' : ''}`));
        button.title = expired ? '原文未在本浏览器保存，请重新上传文档' : '读取仍在有效期内的报告';
        button.addEventListener('click', () => {
          if (expired) { status('此报告已过期，浏览器只保留摘要。请重新上传原文。', true); return; }
          restoreReport(entry.report_id);
        });
        ui.history.append(button);
      });
      ui.clearHistory.disabled = busy || history.length === 0;
    }

    function renderServerReports() {
      ui.reports.replaceChildren();
      ui.reportsCount.textContent = `(${serverReports.length})`;
      if (!serverReports.length) ui.reports.append(make('p', 'muted', '当前没有可恢复的报告。完成一次文档检测后，报告会出现在这里。'));
      serverReports.forEach((entry) => {
        const row = make('div', 'document-report-row');
        const open = make('button', 'document-history-item');
        open.type = 'button'; open.disabled = busy;
        open.append(make('strong', '', entry.filename || '粘贴的长文'));
        open.append(make('span', '', `${dateText(entry.created_at)} · 特征覆盖率 ${percentText(entry.highlighted_total_percent)}`));
        open.append(make('span', '', `有效至 ${dateText(entry.expires_at)}`));
        open.setAttribute('aria-label', `打开报告：${entry.filename || '粘贴的长文'}`);
        open.addEventListener('click', () => restoreReport(entry.report_id));
        const remove = make('button', 'document-report-delete', '删除');
        remove.type = 'button'; remove.disabled = busy;
        remove.setAttribute('aria-label', `删除报告：${entry.filename || '粘贴的长文'}`);
        remove.addEventListener('click', () => removeReport(entry.report_id));
        row.append(open, remove); ui.reports.append(row);
      });
    }

    async function refreshReports() {
      const version = ++listVersion;
      listLoading = true; ui.refreshReports.disabled = true;
      try {
        const response = await fetch('/api/document/reports', {cache: 'no-store', signal: AbortSignal.timeout(5000)});
        if (!response.ok) throw new Error('报告列表更新失败，请稍后刷新。');
        const body = await response.json();
        if (!Array.isArray(body.reports)) throw new Error('报告列表格式异常，请刷新重试。');
        if (version !== listVersion) return;
        serverReports = body.reports.filter(entry => entry && typeof entry.report_id === 'string' && entry.report_id.length <= 128).slice(0, 20);
        ui.reportsStatus.textContent = ''; ui.reportsStatus.classList.remove('error');
        renderServerReports();
      } catch (error) {
        if (version !== listVersion) return;
        ui.reportsStatus.classList.add('error');
        ui.reportsStatus.textContent = error.name === 'TimeoutError' ? '报告列表读取超时，可点击刷新重试。' : error instanceof TypeError ? '报告列表暂未连接到服务，可点击刷新重试。' : error.message;
      } finally {
        if (version === listVersion) { listLoading = false; ui.refreshReports.disabled = busy; }
      }
    }

    function emptyResult(title, message) {
      const empty = make('div', 'empty-result');
      empty.append(make('strong', '', title), make('p', '', message));
      ui.result.replaceChildren(empty);
    }

    async function removeReport(reportId) {
      if (busy) return;
      busy = true; busyAction = 'delete'; ++requestVersion;
      status('正在删除报告…'); updateControls();
      try {
        const response = await fetch(`/api/document/reports/${encodeURIComponent(reportId)}`, {method: 'DELETE', signal: AbortSignal.timeout(10000)});
        if (!response.ok && response.status !== 404 && response.status !== 410) throw new Error(`无法删除报告（${response.status}），请重试。`);
        if (currentReport?.report_id === reportId) {
          currentReport = null;
          emptyResult('报告已删除', '可打开其他报告，或检测新的文档。');
        }
        history = history.filter(entry => entry.report_id !== reportId);
        try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch (_) { /* The server report has still been deleted. */ }
        renderHistory();
        serverReports = serverReports.filter(entry => entry.report_id !== reportId); renderServerReports();
        status('已删除所选报告及其提取原文。');
        await refreshReports();
      } catch (error) {
        status(error.name === 'TimeoutError' ? '删除请求超时，请刷新报告列表核对后重试。' : error instanceof TypeError ? '无法连接检测服务，请启动服务后重试删除。' : error.message, true);
      } finally { busy = false; busyAction = ''; updateControls(); }
    }

    async function downloadReport(report, format) {
      if (busy) return;
      busy = true; busyAction = 'export'; ++requestVersion;
      status('正在准备下载报告…'); updateControls();
      try {
        const response = await fetch(`/api/document/reports/${encodeURIComponent(report.report_id)}/${format}`, {signal: AbortSignal.timeout(30000)});
        if (response.status === 404 || response.status === 410) throw new Error('报告已过期或已删除，请重新检测后下载。');
        if (!response.ok) {
          const error = await response.json().catch(() => ({}));
          throw new Error(typeof error.detail === 'string' ? error.detail : `报告下载失败（${response.status}），请重试。`);
        }
        const blob = await response.blob();
        const disposition = response.headers.get('Content-Disposition') || '';
        const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i);
        let filename = `文档检测报告.${format}`;
        if (encodedName) { try { filename = decodeURIComponent(encodedName[1]); } catch (_) { /* Use the fallback filename. */ } }
        const url = URL.createObjectURL(blob), anchor = make('a');
        anchor.href = url; anchor.download = filename; anchor.hidden = true;
        document.body.append(anchor); anchor.click(); anchor.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        status('报告已准备好，浏览器将开始下载。');
      } catch (error) {
        status(error.name === 'TimeoutError' ? '报告下载超时，请稍后重试。' : error instanceof TypeError ? '无法连接检测服务，请启动服务后重试下载。' : error.message, true);
        await refreshReports();
      } finally { busy = false; busyAction = ''; updateControls(); }
    }

    function updateControls() {
      const characters = codepointCount(ui.text.value);
      ui.count.textContent = `${characters.toLocaleString('zh-CN')} / 100,000 字符`;
      ui.count.style.color = characters > MAX_CHARACTERS ? '#a64c40' : '';
      const usable = mode === 'file' ? selectedFile !== null : ui.text.value.trim().length > 0 && characters <= MAX_CHARACTERS;
      ui.detect.disabled = busy || !usable;
      ui.modeFile.disabled = busy;
      ui.modeText.disabled = busy;
      ui.file.disabled = busy;
      ui.text.disabled = busy;
      ui.clearFile.disabled = busy;
      ui.cancel.hidden = !busy || !controller;
      ui.cancel.disabled = cancelRequested;
      ui.cancel.textContent = activeJobId ? '取消任务' : '取消等待';
      ui.background.hidden = !busy || !activeJobId || busyAction !== 'detect';
      ui.background.disabled = cancelRequested;
      ui.dropzone.classList.toggle('busy', busy);
      ui.detect.querySelector('span').textContent = busy ? ({restore: '正在读取报告…', delete: '正在删除报告…', export: '正在导出报告…'}[busyAction] || '正在提取和分段分析…') : '检测文档并标注';
      ui.history.querySelectorAll('button').forEach((button) => { button.disabled = busy; });
      ui.reports.querySelectorAll('button').forEach((button) => { button.disabled = busy; });
      ui.result.querySelectorAll('.document-exports button').forEach((button) => { button.disabled = busy; });
      ui.clearHistory.disabled = busy || history.length === 0;
      ui.refreshReports.disabled = busy || listLoading;
      ui.saveHistory.disabled = busy;
      ui.jobs.querySelectorAll('button').forEach(button => { button.disabled = busy || button.dataset.cancelling === 'true'; });
    }

    function chooseMode(next) {
      if (busy) return;
      mode = next;
      ui.modeFile.classList.toggle('selected', next === 'file');
      ui.modeText.classList.toggle('selected', next === 'text');
      ui.modeFile.setAttribute('aria-pressed', String(next === 'file'));
      ui.modeText.setAttribute('aria-pressed', String(next === 'text'));
      ui.filePane.hidden = next !== 'file';
      ui.textPane.hidden = next !== 'text';
      status();
      updateControls();
    }

    function chooseFile(file) {
      if (busy || !file) return;
      const extension = file.name.split('.').pop().toLowerCase();
      const rejectFile = (message) => {
        selectedFile = null;
        ui.file.value = '';
        ui.fileMeta.hidden = true;
        ui.name.textContent = '';
        ui.size.textContent = '';
        status(message, true);
        updateControls();
      };
      if (!acceptedExtensions.has(extension)) {
        rejectFile(extension === 'doc' ? '旧版 .doc 暂不支持，请在 Word 中另存为 .docx 后上传。' : '请上传 DOCX、PDF、TXT 或 MD 文件。');
        return;
      }
      if (file.size === 0 || file.size > MAX_BYTES) {
        rejectFile(file.size === 0 ? '文件为空，请重新选择。' : '文件超过 20 MB，请减少文档内容或拆分后上传。');
        return;
      }
      selectedFile = file;
      ui.fileMeta.hidden = false;
      ui.name.textContent = file.name;
      ui.size.textContent = file.size < 1024 * 1024 ? `${(file.size / 1024).toFixed(1)} KB` : `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
      chooseMode('file');
      status();
      updateControls();
    }

    async function responseJson(response) {
      let body;
      try { body = await response.json(); }
      catch (_) { throw new Error(response.ok ? '服务器未返回可读取的检测结果。' : `请求失败（${response.status}），请稍后重试。`); }
      if (!response.ok) {
        const detail = body.detail;
        const message = typeof detail === 'string' ? detail : typeof body.error === 'string' ? body.error : `请求失败（${response.status}）。`;
        throw new Error(message);
      }
      if (!body || typeof body.extracted_text !== 'string' || typeof body.report_id !== 'string') throw new Error('检测结果缺少原文或报告编号，请重新检测。');
      return body;
    }

    function loadingResult() {
      const box = make('div', 'empty-result');
      const progress = make('div', 'busy-state');
      const label = make('span', '', '正在提交文档任务…'); label.id = 'document-task-stage';
      progress.append(make('span', 'spinner'), label);
      const bar = make('progress', 'document-task-progress'); bar.id = 'document-task-progress'; bar.max = 100; bar.value = 0;
      bar.setAttribute('aria-label', '当前文档处理进度');
      box.append(progress, bar, make('p', '', '可在后台继续处理，刷新后从文档任务恢复。'));
      ui.result.replaceChildren(box);
      ui.result.setAttribute('aria-busy', 'true');
    }

    async function detectDocument() {
      if (busy) return;
      const characters = codepointCount(ui.text.value);
      if ((mode === 'file' && !selectedFile) || (mode === 'text' && !ui.text.value.trim())) return;
      if (mode === 'text' && characters > MAX_CHARACTERS) { status('原文超过 100,000 字符，请按章节拆分后检测。', true); return; }
      const form = mode === 'file' ? new FormData() : new URLSearchParams();
      if (mode === 'file') form.append('file', selectedFile);
      else form.append('text', ui.text.value);
      return runDocumentJob(form);
    }

    async function resumeJob(job) {
      if (busy) return;
      return runDocumentJob(null, job);
    }

    async function runDocumentJob(form, existingJob = null) {
      const version = ++requestVersion;
      controller = new AbortController();
      busy = true;
      busyAction = 'detect';
      activeJobId = null; cancelRequested = false; backgrounded = false;
      status(existingJob ? '正在恢复任务进度…' : '正在提交文档任务…');
      loadingResult();
      updateControls();
      try {
        let job = existingJob;
        if (!job) {
          const response = await fetch('/api/document/jobs', {method: 'POST', body: form, signal: controller.signal});
          const body = await response.json();
          if (!response.ok) throw Error(typeof body.detail === 'string' ? body.detail : '文档任务未提交，请重试。');
          if (typeof body.job_id !== 'string') throw Error('未取得文档任务编号，请刷新任务列表。');
          job = body;
        }
        const report = await followJob(job, controller.signal);
        if (version !== requestVersion) return;
        currentReport = report;
        renderReport(report);
        status('检测完成。可查看下划线段落并下载带标注的报告。');
        saveHistory(report);
        refreshReports();
      } catch (error) {
        if (version !== requestVersion) return;
        if (error.name === 'JobCancelled') status('任务已取消，可重新选择文档。');
        else if (error.name === 'AbortError') status(backgrounded ? '任务已转入后台，可从文档任务查看进度和结果。' : '已取消等待，可从文档任务查看处理状态。');
        else if (error.name === 'TimeoutError') status('进度查询超时，可从文档任务恢复。', true);
        else status(error instanceof TypeError ? '无法连接检测服务，请检查本地服务是否启动。' : error.message, true);
        if (currentReport) renderReport(currentReport);
        else {
          const empty = make('div', 'empty-result');
          empty.append(make('strong', '', backgrounded ? '任务已转入后台' : error.name === 'JobCancelled' ? '任务已取消' : '本次检测未完成'), make('p', '', backgrounded ? '完成后可从当前可用报告查看结果。' : '可从文档任务查看状态，或重新提交文档。'));
          ui.result.replaceChildren(empty);
        }
      } finally {
        if (version === requestVersion) {
          busy = false;
          busyAction = '';
          controller = null;
          activeJobId = null; cancelRequested = false; backgrounded = false;
          ui.result.removeAttribute('aria-busy');
          updateControls();
          refreshJobs(); refreshReports();
        }
      }
    }

    async function restoreReport(reportId) {
      if (busy) return;
      const version = ++requestVersion;
      busy = true;
      busyAction = 'restore';
      controller = new AbortController();
      status('正在读取已有报告…');
      updateControls();
      try {
        const response = await fetch(`/api/document/reports/${encodeURIComponent(reportId)}`, {signal: controller.signal});
        if (response.status === 404 || response.status === 410) {
          refreshReports();
          throw new Error('报告已过期或服务已重启，请重新上传文档。');
        }
        const report = await responseJson(response);
        if (version !== requestVersion) return;
        currentReport = report;
        renderReport(report);
        status('已恢复仍在有效期内的报告。');
      } catch (error) {
        if (version !== requestVersion) return;
        status(error.name === 'AbortError' ? '已取消读取报告。' : error instanceof TypeError ? '无法连接检测服务，请启动本地服务后重试。' : error.message, error.name !== 'AbortError');
      } finally {
        if (version === requestVersion) { busy = false; busyAction = ''; controller = null; updateControls(); }
      }
    }

    function metric(label, value, explanation) {
      const card = make('div', 'document-metric');
      card.append(make('strong', '', label), make('div', 'value', value), make('p', '', explanation));
      return card;
    }

    function renderReport(report) {
      const summary = report.document_summary || {};
      const detection = report.detection || {};
      const text = report.extracted_text;
      const points = Array.from(text);
      const normalizedSegments = (Array.isArray(detection.segments) ? detection.segments : [])
        .filter((segment) => segment && Number.isInteger(segment.start) && Number.isInteger(segment.end) && segment.start >= 0 && segment.end > segment.start && segment.end <= points.length)
        .sort((a, b) => a.start - b.start);
      const segments = [];
      let previousEnd = 0;
      for (const segment of normalizedSegments) {
        if (segment.start < previousEnd) continue;
        segments.push(segment);
        previousEnd = segment.end;
      }
      const flagged = segments.filter((segment) => segment.highlighted === true);
      const scoredCount = segments.filter((segment) => finite(segment.risk_score)).length;
      const fragment = document.createDocumentFragment();
      const header = make('div', 'document-result-header');
      const heading = make('div');
      heading.append(make('small', '', '文档检测结果'), make('h3', '', report.filename || '粘贴的长文'));
      header.append(heading, make('span', 'document-complete-label', '自动分析完成'));
      fragment.append(header);
      const metrics = make('div', 'document-metrics');
      metrics.append(metric('AI 特征覆盖率', scoredCount > 0 ? percentText(summary.highlighted_total_percent) : '—', '高风险段字符 / 文档非空白字符'));
      metrics.append(metric('可评分内容覆盖率', percentText(summary.scored_character_percent), '已评分非空白字符 / 全文'));
      metrics.append(metric('可评分段落模型均分', finite(summary.mean_segment_score) ? percentText(summary.mean_segment_score * 100) : '—', '按字符加权；不是 AI 写作比例'));
      fragment.append(metrics);
      fragment.append(make('p', 'document-metric-note', 'AI 特征覆盖率只统计被模型标为高风险的段落，不等于真实的 AI 撰写比例。未评分内容不被视为人类创作。'));
      if (summary.mixed_signals === true) fragment.append(make('div', 'document-warning', '不同段落的模型信号存在明显分歧。这里分别展示覆盖率与段落均分，不将均分解释为整篇文档的 AI 创作概率。'));
      if (scoredCount === 0) fragment.append(make('div', 'document-warning', '未发现符合当前模型范围的可评分段落，因此不给出 AI 特征覆盖率。请查看原文中的未评分说明。'));
      const warnings = Array.isArray(report.warnings) ? report.warnings.filter((item) => typeof item === 'string') : [];
      if (warnings.length) {
        const box = make('div', 'document-warning');
        const list = make('ul');
        warnings.forEach((message) => list.append(make('li', '', message)));
        box.append(list); fragment.append(box);
      }
      const distribution = make('div', 'document-score-distribution');
      distribution.append(make('h4', '', '段落分数分布'));
      const bins = [
        {label: '0–20%', test: (score) => score < .2, color: 'low'},
        {label: '20–80%', test: (score) => score >= .2 && score < .8, color: 'middle'},
        {label: '80–99%', test: (score) => score >= .8 && score < .99, color: 'upper'},
        {label: '99–100%', test: (score) => score >= .99, color: 'high'}
      ];
      const binGrid = make('div', 'document-score-bin-grid');
      bins.forEach((bin) => {
        const count = segments.filter((segment) => finite(segment.risk_score) && bin.test(segment.risk_score)).length;
        const box = make('div', `document-score-bin ${bin.color}`);
        box.append(make('span', '', bin.label), make('strong', '', `${count} 段`));
        const track = make('div', 'document-score-bin-track');
        const fill = make('div', 'document-score-bin-fill');
        fill.style.width = `${scoredCount ? count / scoredCount * 100 : 0}%`;
        track.setAttribute('role', 'meter'); track.setAttribute('aria-label', `分数 ${bin.label} 的段落数量`);
        track.setAttribute('aria-valuemin', '0'); track.setAttribute('aria-valuemax', String(Math.max(scoredCount, 1))); track.setAttribute('aria-valuenow', String(count));
        track.append(fill); box.append(track); binGrid.append(box);
      });
      distribution.append(binGrid, make('p', '', '按已评分段落数量统计；区间与柱长是模型输出的分布，不是作者身份或 AI 写作比例。'));
      fragment.append(distribution);
      const exports = make('div', 'document-exports');
      [['docx', '带下划线的 Word'], ['html', '标注 HTML / 打印 PDF'], ['json', '完整结果 JSON']].forEach(([format, label]) => {
        const button = make('button', 'document-export-link', label);
        button.type = 'button';
        button.addEventListener('click', () => downloadReport(report, format));
        exports.append(button);
      });
      const agentSummary = make('button', 'document-export-link', '交给 Agent 汇总');
      agentSummary.type = 'button';
      agentSummary.addEventListener('click', () => window.dispatchEvent(new CustomEvent('aigc:document-result', {detail: report})));
      exports.append(agentSummary);
      const deleteReport = make('button', 'document-export-link', '删除本次报告');
      deleteReport.type = 'button';
      deleteReport.title = '从本地服务删除本次报告和提取原文，并移除对应浏览器摘要记录';
      deleteReport.addEventListener('click', () => removeReport(report.report_id));
      exports.append(deleteReport);
      fragment.append(exports, make('p', 'document-expiry', `报告${report.expires_at ? `有效至 ${dateText(report.expires_at)}` : '约 30 分钟内有效'}；服务重启后需重新检测。下载文件基于提取后的文字，不是对原文档排版的原样复制。`));

      const readerHeading = make('div', 'document-reader-heading');
      readerHeading.append(make('h4', '', '原文与段落标注'));
      const copy = make('button', 'document-link-button', '复制提取原文');
      copy.type = 'button';
      copy.addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(text); copy.textContent = '已复制'; setTimeout(() => { copy.textContent = '复制提取原文'; }, 2000); }
        catch (_) { status('浏览器不允许直接复制，请在原文区域选择文字后复制。', true); }
      });
      readerHeading.append(copy); fragment.append(readerHeading);
      const controls = make('div', 'document-reader-controls');
      const annotationLabel = make('label');
      const annotationToggle = make('input'); annotationToggle.type = 'checkbox'; annotationToggle.checked = true;
      annotationLabel.append(annotationToggle, document.createTextNode('显示高风险下划线'));
      const unscoredLabel = make('label'); const unscoredToggle = make('input'); unscoredToggle.type = 'checkbox';
      unscoredLabel.append(unscoredToggle, document.createTextNode('显示未评分段落'));
      controls.append(annotationLabel, unscoredLabel); fragment.append(controls);
      const legend = make('div', 'document-reader-legend');
      legend.append(make('span', 'risk-key', '实线：高风险段落'), make('span', 'unscored-key', '虚线：未评分（可选）'), make('span', '', '选中段落查看依据'));
      fragment.append(legend);
      const reader = make('div', 'document-reader'); reader.id = 'document-annotated-text';
      reader.setAttribute('aria-label', '文档提取原文，高风险段落以实线下划线标记');
      reader.setAttribute('tabindex', '0');
      const detail = make('div', 'document-segment-detail', '点击或用键盘选中段落，可查看模型分数、来源位置与评分依据。');
      detail.setAttribute('aria-live', 'polite');
      const sectionLabel = (segment) => {
        const sections = Array.isArray(report.sections) ? report.sections : [];
        const section = sections.find((entry) => entry && Number.isInteger(entry.start) && Number.isInteger(entry.end) && entry.start <= segment.start && entry.end > segment.start);
        return section && typeof section.label === 'string' ? section.label : section && finite(section.page) ? `第 ${section.page} 页` : '';
      };
      const describe = (segment, index) => {
        const location = sectionLabel(segment);
        const score = finite(segment.risk_score) ? `模型分数 ${percentText(segment.risk_score * 100)}` : '当前段落未评分';
        const flag = segment.highlighted === true ? '高风险下划线段落' : finite(segment.risk_score) ? '未达到高风险标注阈值' : '不属于当前模型可评分范围';
        return `第 ${index + 1} 段${location ? ` · ${location}` : ''} · ${score} · ${flag}`;
      };
      let focused = null;
      function showSegment(segment, index, span) {
        if (focused) focused.classList.remove('is-focused');
        focused = span; span.classList.add('is-focused');
        detail.replaceChildren(make('strong', '', describe(segment, index)));
        const evidence = Array.isArray(segment.evidence) ? segment.evidence.map(evidenceText).filter(Boolean) : [];
        if (evidence.length) {
          const list = make('ul'); evidence.slice(0, 8).forEach((entry) => list.append(make('li', '', entry))); detail.append(list);
        } else detail.append(make('span', '', '该分数来自段落分类模型；下划线覆盖整段，不代表逐词确认来源。'));
      }
      let cursor = 0;
      const spanNodes = [];
      segments.forEach((segment, index) => {
        if (segment.start > cursor) reader.append(document.createTextNode(points.slice(cursor, segment.start).join('')));
        const span = make('span', `document-span${segment.highlighted === true ? ' document-risk-span' : ''}${!finite(segment.risk_score) ? ' document-unscored-span' : ''}`, points.slice(segment.start, segment.end).join(''));
        span.id = `document-segment-${index}`;
        span.tabIndex = 0;
        span.title = describe(segment, index);
        span.setAttribute('aria-label', describe(segment, index));
        span.addEventListener('click', () => showSegment(segment, index, span));
        span.addEventListener('focus', () => showSegment(segment, index, span));
        span.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); showSegment(segment, index, span); }
        });
        reader.append(span); spanNodes.push(span); cursor = segment.end;
      });
      if (cursor < points.length) reader.append(document.createTextNode(points.slice(cursor).join('')));
      annotationToggle.addEventListener('change', () => reader.classList.toggle('hide-annotations', !annotationToggle.checked));
      unscoredToggle.addEventListener('change', () => reader.classList.toggle('show-unscored', unscoredToggle.checked));
      fragment.append(reader, detail);
      const listHeading = make('div', 'document-segment-list-header');
      listHeading.append(make('h4', '', `高风险段落 ${flagged.length} 个`));
      const navigation = make('div', 'document-navigation');
      const previous = make('button', '', '上一段'); const next = make('button', '', '下一段');
      previous.type = 'button'; next.type = 'button'; previous.disabled = !flagged.length; next.disabled = !flagged.length;
      navigation.append(previous, next); listHeading.append(navigation); fragment.append(listHeading);
      const list = make('div', 'document-segment-list');
      let flaggedPosition = -1;
      const flaggedIndexes = segments.map((segment, index) => segment.highlighted === true ? index : -1).filter((index) => index >= 0);
      function focusFlag(position) {
        if (!flaggedIndexes.length) return;
        flaggedPosition = (position + flaggedIndexes.length) % flaggedIndexes.length;
        const index = flaggedIndexes[flaggedPosition]; const span = spanNodes[index];
        span.scrollIntoView({block: 'nearest', inline: 'nearest', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth'});
        span.focus({preventScroll: true}); showSegment(segments[index], index, span);
      }
      previous.addEventListener('click', () => focusFlag(flaggedPosition <= 0 ? flaggedIndexes.length - 1 : flaggedPosition - 1));
      next.addEventListener('click', () => focusFlag(flaggedPosition + 1));
      if (!flagged.length) list.append(make('p', 'muted', scoredCount ? '没有段落达到高风险标注阈值；这不证明全文由人类创作。' : '当前文档没有可评分段落，无法定位疑似 AI 特征。'));
      flaggedIndexes.forEach((index, position) => {
        const segment = segments[index]; const button = make('button', 'document-segment-jump'); button.type = 'button';
        const excerpt = points.slice(segment.start, Math.min(segment.end, segment.start + 70)).join('').replace(/\s+/g, ' ').trim();
        button.append(make('span', '', `第 ${index + 1} 段${sectionLabel(segment) ? ` · ${sectionLabel(segment)}` : ''}：${excerpt}${segment.end - segment.start > 70 ? '…' : ''}`), make('strong', '', finite(segment.risk_score) ? percentText(segment.risk_score * 100) : '—'));
        button.addEventListener('click', () => focusFlag(position)); list.append(button);
      });
      fragment.append(list);
      const meta = make('div', 'document-source-meta');
      meta.append(make('span', '', `全文 ${points.length.toLocaleString('zh-CN')} 字符 · ${scoredCount} 个段落已评分 · ${segments.length - scoredCount} 个段落未评分 · 高风险字符 / 可评分字符 ${percentText(summary.highlighted_character_percent)}`));
      meta.append(make('span', '', `检测时间 ${dateText(report.created_at)} · ${report.format || 'text'} · 当前分数尚未经过真实平台上线验收`));
      if (typeof report.source_sha256 === 'string') meta.append(make('span', 'document-source-hash', `来源 SHA-256：${report.source_sha256}`));
      fragment.append(meta);
      ui.result.replaceChildren(fragment);
      ui.result.removeAttribute('aria-busy');
      icons();
    }

    ui.modeFile.addEventListener('click', () => chooseMode('file'));
    ui.modeText.addEventListener('click', () => chooseMode('text'));
    ui.file.addEventListener('change', () => chooseFile(ui.file.files[0]));
    ui.dropzone.tabIndex = 0;
    ui.dropzone.setAttribute('role', 'button');
    ui.dropzone.addEventListener('click', (event) => { if (busy) event.preventDefault(); });
    ui.dropzone.addEventListener('keydown', (event) => {
      if ((event.key === 'Enter' || event.key === ' ') && !busy) { event.preventDefault(); ui.file.click(); }
    });
    ui.dropzone.addEventListener('dragover', (event) => { event.preventDefault(); if (!busy) ui.dropzone.classList.add('dragging'); });
    ui.dropzone.addEventListener('dragleave', () => ui.dropzone.classList.remove('dragging'));
    ui.dropzone.addEventListener('drop', (event) => {
      event.preventDefault(); ui.dropzone.classList.remove('dragging');
      if (!busy && event.dataTransfer && event.dataTransfer.files.length) {
        if (event.dataTransfer.files.length > 1) { status('一次检测一份文档，请只拖入一个文件。', true); return; }
        chooseFile(event.dataTransfer.files[0]);
      }
    });
    ui.clearFile.addEventListener('click', () => {
      if (busy) return;
      selectedFile = null; ui.file.value = ''; ui.fileMeta.hidden = true; ui.name.textContent = ''; ui.size.textContent = ''; status(); updateControls();
    });
    ui.text.addEventListener('input', updateControls);
    ui.detect.addEventListener('click', detectDocument);
    ui.cancel.addEventListener('click', () => { if (activeJobId) requestCancel(activeJobId); else if (controller) controller.abort(); });
    ui.background.addEventListener('click', () => { if (activeJobId && controller) { backgrounded = true; controller.abort(); } });
    ui.refreshJobs.addEventListener('click', refreshJobs);
    ui.saveHistory.addEventListener('change', () => { if (ui.saveHistory.checked && currentReport) saveHistory(currentReport); });
    ui.clearHistory.addEventListener('click', () => {
      if (busy) return;
      try { localStorage.removeItem(HISTORY_KEY); history = []; renderHistory(); status('本浏览器的检测记录已清空。'); }
      catch (_) { status('浏览器未允许清除存储，可在浏览器设置中移除此站点的数据。', true); }
    });
    window.addEventListener('aigc:open-document-report', (event) => {
      const reportId = event.detail?.report_id;
      if (typeof reportId === 'string' && reportId.length > 0 && reportId.length <= 128) restoreReport(reportId);
    });
    window.AIGCDocument = Object.freeze({renderInline, showReport: (report) => {
      if (!report || typeof report.extracted_text !== 'string' || typeof report.report_id !== 'string' || report.report_id.length > 128) return false;
      if (busy) { status('当前文档仍在分析，请稍后打开其他报告。'); return false; }
      currentReport = report;
      renderReport(report);
      status('已打开文档检测报告。');
      saveHistory(report);
      refreshReports();
      return true;
    }});
    ui.refreshReports.addEventListener('click', () => { if (!busy) refreshReports(); });
    const scheduleRefresh = () => {
      clearTimeout(listTimer);
      clearTimeout(jobsTimer);
      if (!document.hidden && byId('view-document').classList.contains('active')) {
        refreshJobs();
        if (!busy) refreshReports();
        listTimer = setTimeout(scheduleRefresh, 30000);
      }
    };
    window.addEventListener('aigc:view-changed', scheduleRefresh);
    document.addEventListener('visibilitychange', scheduleRefresh);
    renderHistory(); updateControls(); refreshReports(); refreshJobs();
    if (byId('view-document').classList.contains('active')) listTimer = setTimeout(scheduleRefresh, 30000);
  });
})();
