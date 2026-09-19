const browserApi = globalThis.browser ?? globalThis.chrome;
const attached = new WeakSet();
const menuAttached = new WeakSet();

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
    pageUrl: location.href,
    videoUrl: selectedUrl(video),
    title: document.title,
    scale: 2,
    limitSeconds: isPreview ? 5 : null
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
    overlay.addEventListener("ended", () => {
      video.style.visibility = "";
      overlay.remove();
    }, { once: true });
    return true;
  } catch (error) {
    overlay.remove();
    return false;
  }
}

function attach(video) {
  if (attached.has(video)) return;
  attached.add(video);
  const control = document.createElement("div");
  control.className = "vu-action";
  control.innerHTML = '<span>Melhorar vídeo</span><button type="button">Prévia 2×</button><button type="button">2× completo</button>';
  (document.body || document.documentElement).append(control);

  const update = () => place(control, video);
  new ResizeObserver(update).observe(video);
  window.addEventListener("scroll", update, { passive: true });
  update();

  control.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      const isPreview = button.textContent.includes("Prévia");
      const buttons = control.querySelectorAll("button");
      buttons.forEach((item) => { item.disabled = true; });
      button.textContent = "Enviando…";
      const reply = await browserApi.runtime.sendMessage({
        type: "UPSCALE_VIDEO",
        pipeline: "hybrid-streaming",
        pageUrl: location.href,
        videoUrl: selectedUrl(video),
        title: document.title,
        scale: 2,
        limitSeconds: isPreview ? 5 : null
      });
      if (reply?.ok) {
        const output = reply.result?.output;
        if (output && await replaceVideoSource(video, output)) {
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
      } else {
        button.textContent = reply?.error
          ? `Erro: ${reply.error.slice(0, 48)}`
          : "Instale o componente local";
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
