// Experimental client-side upscaling backend.
//
// This module never leaves the page: it re-renders the already-decoded
// <video> element onto a larger canvas (WebGPU when available, Canvas 2D
// otherwise), captures the canvas plus the original audio track, and
// records the result with MediaRecorder. It is a resample pipeline, not a
// neural super-resolution model — it exists to validate a fully in-browser
// "Camada 3" alternative to the native Real-ESRGAN backend, following the
// same frame-transport shape already used by streaming-engine/. A future
// WebGPU compute shader (or WebNN/ONNX Runtime Web model) can replace the
// sampler in upscaleFrameWebGPU without touching the rest of the pipeline.

const WGSL_SHADER = `
  struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
  };

  @vertex
  fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
      vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
    );
    var uvs = array<vec2<f32>, 6>(
      vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
      vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[index], 0.0, 1.0);
    out.uv = uvs[index];
    return out;
  }

  @group(0) @binding(0) var frameSampler: sampler;
  @group(0) @binding(1) var frameTexture: texture_external;

  @fragment
  fn fragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSampleBaseClampToEdge(frameTexture, frameSampler, in.uv);
  }
`;

function pickMimeType() {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported?.(type)) || "video/webm";
}

async function createWebGpuRenderer(canvas) {
  if (!("gpu" in navigator)) return null;
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  if (!adapter) return null;
  const device = await adapter.requestDevice().catch(() => null);
  if (!device) return null;

  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const module = device.createShaderModule({ code: WGSL_SHADER });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vertexMain" },
    fragment: { module, entryPoint: "fragmentMain", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

  return {
    kind: "webgpu",
    draw(videoFrame) {
      const externalTexture = device.importExternalTexture({ source: videoFrame });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: externalTexture },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
  };
}

function createCanvas2DRenderer(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return {
    kind: "canvas2d",
    draw(source) {
      ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    },
  };
}

/**
 * Renders `video` onto an upscaled canvas and records the result with
 * MediaRecorder, reusing the original audio track. Resolves with a Blob URL
 * that can be assigned directly to a new <video> element.
 *
 * @param {HTMLVideoElement} video
 * @param {{scale?: number, limitSeconds?: number|null, onProgress?: (seconds:number)=>void}} options
 */
export async function upscaleInBrowser(video, options = {}) {
  const scale = options.scale === 4 ? 4 : 2;
  const limitSeconds = options.limitSeconds || null;
  const width = Math.max(2, Math.round((video.videoWidth || video.clientWidth) * scale));
  const height = Math.max(2, Math.round((video.videoHeight || video.clientHeight) * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const renderer = (await createWebGpuRenderer(canvas)) || createCanvas2DRenderer(canvas);
  const useWebCodecs = renderer.kind === "webgpu" && "VideoFrame" in window;

  const fps = 30;
  const canvasStream = canvas.captureStream(fps);
  const originalStream = typeof video.captureStream === "function" ? video.captureStream() : null;
  const audioTracks = originalStream ? originalStream.getAudioTracks() : [];
  const outputStream = new MediaStream([...canvasStream.getVideoTracks(), ...audioTracks]);

  const recorder = new MediaRecorder(outputStream, { mimeType: pickMimeType() });
  const chunks = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };

  const finished = new Promise((resolve, reject) => {
    recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType }));
    recorder.onerror = (event) => reject(event.error || new Error("MediaRecorder failed"));
  });

  let stop = false;
  const startedAt = video.currentTime;
  const stepWithVideoFrameCallback = () => new Promise((resolve) => {
    video.requestVideoFrameCallback(() => resolve());
  });

  recorder.start();
  try {
    while (!stop) {
      if (useWebCodecs) {
        const frame = new VideoFrame(video);
        renderer.draw(frame);
        frame.close();
      } else {
        renderer.draw(video);
      }
      options.onProgress?.(video.currentTime - startedAt);
      if (video.ended) break;
      if (limitSeconds && video.currentTime - startedAt >= limitSeconds) break;
      await stepWithVideoFrameCallback();
    }
  } finally {
    stop = true;
    recorder.stop();
    canvasStream.getTracks().forEach((track) => track.stop());
  }

  const blob = await finished;
  return URL.createObjectURL(blob);
}

export function browserPipelineSupported() {
  return typeof MediaRecorder !== "undefined" && typeof HTMLCanvasElement.prototype.captureStream === "function";
}
