const browserApi = globalThis.browser ?? globalThis.chrome;
const HOST_NAME = "com.local_video_upscaler";

browserApi.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type !== "UPSCALE_VIDEO") return false;

  browserApi.runtime.sendNativeMessage(HOST_NAME, message)
    .then((result) => {
      if (result?.error) {
        console.error("Video Upscaler Native Host failed:", result.error);
        sendResponse({ ok: false, error: result.error });
        return;
      }
      sendResponse({ ok: true, result });
    })
    .catch((error) => {
      const detail = error?.message || String(error);
      console.error("Video Upscaler Native Host error:", detail);
      sendResponse({ ok: false, error: detail });
    });
  return true;
});
