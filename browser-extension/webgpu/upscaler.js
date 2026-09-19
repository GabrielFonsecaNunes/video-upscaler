(() => {
  const shader = `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var source: texture_2d<f32>;
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
  let size = vec2f(textureDimensions(source));
  let pixel = 1.0 / size;
  let center = textureSample(source, filtering, input.uv);
  let blur = (
    textureSample(source, filtering, input.uv + vec2f(pixel.x, 0.0)) +
    textureSample(source, filtering, input.uv - vec2f(pixel.x, 0.0)) +
    textureSample(source, filtering, input.uv + vec2f(0.0, pixel.y)) +
    textureSample(source, filtering, input.uv - vec2f(0.0, pixel.y))
  ) * 0.25;
  return vec4f(clamp(center + (center - blur) * 0.18, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

  class WebGPUUpscaler {
    constructor(video) {
      this.video = video;
      this.resize = this.resize.bind(this);
      this.canvas = document.createElement("canvas");
      this.canvas.className = "vu-webgpu-canvas";
      this.canvas.style.cssText = "position:absolute;z-index:2147483646;object-fit:contain;background:#000";
      this.running = false;
      this.ready = this.initialize();
    }

    async initialize() {
      if (!navigator.gpu) throw new Error("WebGPU não está disponível neste navegador");
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error("GPU não encontrada");
      this.device = await adapter.requestDevice();
      this.context = this.canvas.getContext("webgpu");
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      this.pipeline = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module: this.device.createShaderModule({ code: shader }), entryPoint: "vertex" },
        fragment: {
          module: this.device.createShaderModule({ code: shader }),
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
      this.canvas.style.top = `${window.scrollY + rect.top}px`;
      this.canvas.style.left = `${window.scrollX + rect.left}px`;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
    }

    async start() {
      await this.ready;
      this.resize();
      (document.body || document.documentElement).append(this.canvas);
      this.video.style.visibility = "hidden";
      this.running = true;
      this.frame();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.video);
      window.addEventListener("scroll", this.resize, { passive: true });
      return true;
    }

    frame() {
      if (!this.running) return;
      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        if (!this.texture || this.sourceWidth !== this.video.videoWidth || this.sourceHeight !== this.video.videoHeight) {
          this.texture?.destroy();
          this.sourceWidth = this.video.videoWidth;
          this.sourceHeight = this.video.videoHeight;
          this.texture = this.device.createTexture({
            size: [this.video.videoWidth, this.video.videoHeight],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
          });
          this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: this.texture.createView() },
              { binding: 1, resource: this.sampler },
            ],
          });
        }
        this.device.queue.copyExternalImageToTexture(
          { source: this.video },
          { texture: this.texture },
          [this.video.videoWidth, this.video.videoHeight],
        );
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
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
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
