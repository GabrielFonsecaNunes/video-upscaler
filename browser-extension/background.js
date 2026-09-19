const browserApi = globalThis.browser ?? globalThis.chrome;
const HOST_NAME = "com.local_video_upscaler";

browserApi.runtime.onMessage.addListener((message) => {
  if (message.type !== "UPSCALE_VIDEO") return undefined;

  return browserApi.runtime.sendNativeMessage(HOST_NAME, message)
    .then((result) => ({ ok: true, result }))
    .catch((error) => {
      const detail = error?.message || String(error);
      console.error("Video Upscaler Native Host error:", detail);
      return { ok: false, error: detail };
    });
});
