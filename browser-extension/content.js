const browserApi = globalThis.browser ?? globalThis.chrome;
const attached = new WeakSet();
const menuAttached = new WeakSet();

// Lazily loaded so pages that never click "Processar no navegador" pay no
// WebGPU/MediaRecorder setup cost. See browser-pipeline.js for the backend.
let browserPipelinePromise = null;
function loadBrowserPipeline() {
  browserPipelinePromise ??= import(browserApi.runtime.getURL("browser-pipeline.js"));
  return browserPipelinePromise;
}

if (window.top !== window) {
  document.documentElement.dataset.videoUpscalerFrame = "true";
}

function selectedUrl(video) {
  // YouTube's page URL is more durable than its short-lived streaming URL.
  if (location.hostname.endsWith("youtube.com") || location.hostname === "youtu.be") return location.href;
  return video.currentSrc || video.src || location.href;
}

function place(control, video) {
  const rect = video.getBoundingClientRect();
  control.style.top = `${Math.max(8, window.scrollY + rect.top + 8)}px`;
  control.style.left = `${Math.max(8, window.scrollX + rect.left + 8)}px`;
  control.hidden = rect.width < 160 || rect.height < 90;
}

async function processVideo(video, isPreview) {
  const reply = await browserApi.runtime.sendMessage({
    type: "UPSCALE_VIDEO",
    pipeline: "hybrid-streaming",
    model: "x2plus",
    pageUrl: location.href,
    videoUrl: selectedUrl(video),
    title: document.title,
    scale: 2,
    limitSeconds: isPreview ? 5 : null,
    // Lets the native host route x2plus through streaming-engine/ (frame-by-frame,
    // no PNG sequence) instead of only the direct MLX call. See video_upscaler_host.py.
    enableNativeEngine: true
  });
  if (!reply?.ok || !reply.result?.output) {
    throw new Error(reply?.error || "O processamento local falhou.");
  }
  return replaceVideoSource(video, reply.result.output);
}

function addYouTubeQualityOption(video) {
  const menu = document.querySelector(".ytp-settings-menu");
  if (!menu || menuAttached.has(menu)) return;
  menuAttached.add(menu);
  const item = document.createElement("div");
  item.className = "ytp-menuitem vu-youtube-option";
  item.setAttribute("role", "menuitem");
  item.tabIndex = 0;
  item.innerHTML = '<div class="ytp-menuitem-label">Video Upscaler — 2×</div>';
  item.addEventListener("click", async () => {
    item.setAttribute("aria-disabled", "true");
    item.querySelector(".ytp-menuitem-label").textContent = "Processando vídeo…";
    let success = false;
    try {
      success = await processVideo(video, true);
    } catch {
      success = false;
    }
    item.querySelector(".ytp-menuitem-label").textContent = success
      ? "Video Upscaler — reproduzindo"
      : "Video Upscaler — erro";
  });
  menu.append(item);
}

async function replaceVideoSource(video, outputPath) {
  const clean = String(outputPath || "").trim();
  if (!clean) return false;
  const overlay = document.createElement("video");
  overlay.className = "vu-result-video";
  overlay.controls = true;
  overlay.autoplay = true;
  overlay.playsInline = true;
  overlay.src = clean;
  overlay.style.cssText = "position:absolute;z-index:2147483646;object-fit:contain;background:#000";

  const rect = video.getBoundingClientRect();
  overlay.style.top = `${window.scrollY + rect.top}px`;
  overlay.style.left = `${window.scrollX + rect.left}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  (document.body || document.documentElement).append(overlay);

  try {
    await new Promise((resolve, reject) => {
      overlay.addEventListener("canplay", resolve, { once: true });
      overlay.addEventListener("error", reject, { once: true });
      overlay.load();
    });
    video.pause();
    video.style.visibility = "hidden";
    return true;
  } catch (error) {
    overlay.remove();
    return false;
  }
}

async function runNativePipeline(video, model, isPreview) {
  const reply = await browserApi.runtime.sendMessage({
    type: "UPSCALE_VIDEO",
    pipeline: "hybrid-streaming",
    model,
    pageUrl: location.href,
    videoUrl: selectedUrl(video),
    title: document.title,
    scale: 2,
    limitSeconds: isPreview ? 5 : null,
    // See video_upscaler_host.py: only applies to the x2plus MLX path, and is
    // ignored otherwise (ONNX/CoreML/NCNN paths do not read this flag).
    enableNativeEngine: true
  });
  if (!reply?.ok) {
    throw new Error(reply?.error || "Instale o componente local");
  }
  const output = reply.result?.output;
  if (!output) throw new Error("Resposta local sem vídeo processado.");
  return output;
}

// Experimental: renders and encodes the upscaled video entirely in this tab
// (WebGPU/Canvas2D + MediaRecorder), without the native host or FFmpeg. See
// browser-pipeline.js. Useful when the native companion is not installed.
async function runBrowserPipeline(video, isPreview) {
  const { upscaleInBrowser, browserPipelineSupported } = await loadBrowserPipeline();
  if (!browserPipelineSupported()) {
    throw new Error("Este navegador não suporta o pipeline embutido.");
  }
  return upscaleInBrowser(video, { scale: 2, limitSeconds: isPreview ? 5 : null });
}

function attach(video) {
  if (attached.has(video)) return;
  attached.add(video);
  const control = document.createElement("div");
  control.className = "vu-action";
  control.innerHTML = '<span>Melhorar vídeo</span><label>Modelo <select class="vu-model"><option value="x2plus">Real-ESRGAN x2 (rápido)</option><option value="animevideo">AnimeVideo x4</option><option value="general">General x4</option><option value="x4plus">Real-ESRGAN x4</option><option value="anime_6B">Anime x4 (qualidade)</option><option value="efrlfn_x2" title="~7x mais rápido, qualidade mais suave">EfRLFN x2 (turbo)</option><option value="efrlfn_x4" title="~5x mais rápido, qualidade mais suave">EfRLFN x4 (turbo)</option><option value="rlfn_x2" title="~6x mais rápido, qualidade próxima ao Real-ESRGAN">RLFN x2 (turbo nítido)</option><option value="rlfn_x4" title="~6x mais rápido, qualidade próxima ao Real-ESRGAN">RLFN x4 (turbo nítido)</option><option value="nanovsr_x4" title="~9x mais rápido, usa vários frames (VSR temporal)">NanoVSR x4 (turbo temporal)</option></select></label><button type="button" data-pipeline="native">Prévia</button><button type="button" data-pipeline="native">Processar</button><button type="button" data-pipeline="browser" title="Experimental: processa nesta aba, sem o componente local">No navegador</button>';
  (document.body || document.documentElement).append(control);

  const update = () => place(control, video);
  new ResizeObserver(update).observe(video);
  window.addEventListener("scroll", update, { passive: true });
  update();

  control.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      const isPreview = button.textContent.includes("Prévia");
      const useBrowserPipeline = button.dataset.pipeline === "browser";
      const model = control.querySelector(".vu-model").value;
      const buttons = control.querySelectorAll("button");
      buttons.forEach((item) => { item.disabled = true; });
      button.textContent = useBrowserPipeline ? "Processando na aba…" : "Enviando…";
      try {
        const output = useBrowserPipeline
          ? await runBrowserPipeline(video, isPreview)
          : await runNativePipeline(video, model, isPreview);
        if (await replaceVideoSource(video, output)) {
          button.textContent = "Atualizado";
          const link = document.createElement("a");
          link.href = output;
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = "Abrir vídeo local";
          link.className = "vu-result-link";
          control.append(link);
          setTimeout(() => { control.remove(); }, 2000);
          return;
        }
        button.textContent = "Concluído";
      } catch (error) {
        button.textContent = `Erro: ${String(error?.message || error).slice(0, 48)}`;
      }
      setTimeout(() => { control.remove(); }, 2400);
    });
  });
}

function scan() {
  document.querySelectorAll("video").forEach((video) => {
    attach(video);
    addYouTubeQualityOption(video);
  });
}

const observerRoot = document.documentElement;
if (observerRoot) {
  new MutationObserver(scan).observe(observerRoot, { childList: true, subtree: true });
}
scan();
setInterval(scan, 2000);
