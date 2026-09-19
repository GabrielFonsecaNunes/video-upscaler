(() => {
  const shader = `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var source: texture_external;
@group(0) @binding(1) var filtering: sampler;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
  var positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
  );
  var uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
    vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
  );
  return VertexOutput(vec4f(positions[index], 0.0, 1.0), uvs[index]);
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
  let center = textureSample(source, filtering, input.uv);
  return center;
}
`;

  class WebGPUUpscaler {
    constructor(video) {
      this.video = video;
      this.resize = this.resize.bind(this);
      this.canvas = document.createElement("canvas");
      this.canvas.className = "vu-webgpu-canvas";
      this.canvas.style.cssText = "position:fixed;z-index:2147483646;object-fit:contain;background:#000;pointer-events:none";
      this.running = false;
      this.rendered = false;
      this.ready = this.initialize();
    }

    async initialize() {
      if (!window.isSecureContext) throw new Error("WebGPU exige contexto seguro");
      if (!navigator.gpu) throw new Error("WebGPU não está disponível neste navegador");
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("GPU não encontrada");
      this.device = await adapter.requestDevice();
      this.device.lost.then((info) => {
        this.running = false;
        this.video.style.visibility = "";
        this.canvas.dataset.error = `GPU perdida: ${info.message || info.reason}`;
      });
      this.context = this.canvas.getContext("webgpu");
      if (!this.context) throw new Error("Canvas WebGPU indisponível");
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      const module = this.device.createShaderModule({ code: shader });
      const compilation = await module.getCompilationInfo();
      const error = compilation.messages.find((message) => message.type === "error");
      if (error) throw new Error(`Shader WGSL: ${error.message}`);
      this.pipeline = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: {
          module,
          entryPoint: "fragment",
          targets: [{ format: this.format }],
        },
        primitive: { topology: "triangle-list" },
      });
      this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear" });
    }

    resize() {
      const rect = this.video.getBoundingClientRect();
      const width = Math.max(1, Math.floor(this.video.videoWidth || rect.width));
      const height = Math.max(1, Math.floor(this.video.videoHeight || rect.height));
      this.canvas.width = width * 2;
      this.canvas.height = height * 2;
      this.canvas.style.top = `${rect.top}px`;
      this.canvas.style.left = `${rect.left}px`;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
    }

    async start() {
      await this.ready;
      if (this.video.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise((resolve) => {
          this.video.addEventListener("loadedmetadata", resolve, { once: true });
        });
      }
      this.resize();
      this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      (document.body || document.documentElement).append(this.canvas);
      this.running = true;
      this.firstFrame = new Promise((resolve, reject) => {
        this.resolveFirstFrame = resolve;
        this.rejectFirstFrame = reject;
        setTimeout(() => reject(new Error("WebGPU não recebeu um frame em 5 segundos")), 5000);
      });
      this.frame();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.video);
      window.addEventListener("scroll", this.resize, { passive: true });
      await this.firstFrame;
      return true;
    }

    frame() {
      if (!this.running) return;
      const width = this.video.videoWidth;
      const height = this.video.videoHeight;
      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && width > 0 && height > 0) {
        try {
          const external = this.device.importExternalTexture({ source: this.video });
          const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: external },
              { binding: 1, resource: this.sampler },
            ],
          });
          const encoder = this.device.createCommandEncoder();
          const pass = encoder.beginRenderPass({
            colorAttachments: [{
              view: this.context.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "clear",
              storeOp: "store",
            }],
          });
          pass.setPipeline(this.pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.draw(6);
          pass.end();
          this.device.queue.submit([encoder.finish()]);
          if (!this.rendered) {
            this.device.queue.onSubmittedWorkDone().then(() => {
              if (!this.running || this.rendered) return;
              this.rendered = true;
              this.video.style.visibility = "hidden";
              this.resolveFirstFrame?.();
            });
          }
        } catch (error) {
          this.canvas.dataset.error = error instanceof Error ? error.message : String(error);
          this.video.style.visibility = "";
          this.rejectFirstFrame?.(error);
        }
      }
      requestAnimationFrame(() => this.frame());
    }

    stop() {
      this.running = false;
      this.resizeObserver?.disconnect();
      window.removeEventListener("scroll", this.resize);
      this.canvas.remove();
      this.video.style.visibility = "";
    }
  }

  globalThis.VideoUpscalerWebGPU = WebGPUUpscaler;
})();
