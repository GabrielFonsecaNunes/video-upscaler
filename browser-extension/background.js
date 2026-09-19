const browserApi = globalThis.browser ?? globalThis.chrome;
const HOST_NAME = "com.local_video_upscaler";

browserApi.runtime.onMessage.addListener((message) => {
  if (message.type !== "UPSCALE_VIDEO") return undefined;

  return browserApi.runtime.sendNativeMessage(HOST_NAME, message)
    .then((result) => result?.error
      ? { ok: false, error: result.error }
      : { ok: true, result })
    .catch((error) => {
      const detail = error.message || String(error);
      return { ok: false, error: detail };
    });
});
