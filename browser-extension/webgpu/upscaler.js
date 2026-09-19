(() => {
  const displayShader = `
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
  return textureSampleBaseClampToEdge(source, filtering, input.uv);
}
`;

  const convertShader = `
@group(0) @binding(0) var source: texture_external;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>;
@group(0) @binding(3) var filtering: sampler;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.x || id.y >= params.y) { return; }
  let rgb = textureSampleBaseClampToEdge(source, filtering, vec2f(
    (f32(id.x) + 0.5) / f32(params.x), (f32(id.y) + 0.5) / f32(params.y)));
  let base = (id.y * params.x + id.x) * 3u;
  output[base] = rgb.r; output[base + 1u] = rgb.g; output[base + 2u] = rgb.b;
}
`;

  const convShader = `
struct Params { width: u32, height: u32, inC: u32, outC: u32, activate: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> slope: array<f32>;
@group(0) @binding(5) var<uniform> p: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.width || id.y >= p.height || id.z >= p.outC) { return; }
  var value = bias[id.z];
  for (var ky: i32 = -1; ky <= 1; ky++) {
    for (var kx: i32 = -1; kx <= 1; kx++) {
      let xx = i32(id.x) + kx; let yy = i32(id.y) + ky;
      if (xx < 0 || yy < 0 || xx >= i32(p.width) || yy >= i32(p.height)) { continue; }
      for (var ic: u32 = 0u; ic < p.inC; ic++) {
        let inputIndex = (u32(yy) * p.width + u32(xx)) * p.inC + ic;
        let weightIndex = (((id.z * 3u + u32(ky + 1)) * 3u + u32(kx + 1)) * p.inC) + ic;
        value += input[inputIndex] * weights[weightIndex];
      }
    }
  }
  if (p.activate == 1u && value < 0.0) { value *= slope[id.z]; }
  output[(id.y * p.width + id.x) * p.outC + id.z] = value;
}
`;

  const shuffleShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@group(0) @binding(3) var<storage, read> original: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let width = params.x; let height = params.y; let channels = params.z;
  let x = id.x; let y = id.y;
  if (x >= width * 4u || y >= height * 4u) { return; }
  let sx = x / 4u; let sy = y / 4u; let ox = x % 4u; let oy = y % 4u;
  let channel = oy * 4u + ox;
  let base = (sy * width + sx) * channels + channel * 3u;
  let originalBase = (sy * width + sx) * 3u;
  let rgb = clamp(
    vec3f(input[base], input[base + 1u], input[base + 2u]) +
      vec3f(original[originalBase], original[originalBase + 1u], original[originalBase + 2u]),
    vec3f(0.0), vec3f(1.0));
  textureStore(output, vec2u(x, y), vec4f(rgb, 1.0));
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
      this.inFlight = 0;
      this.frameSequence = 0;
      this.lastDisplayedSequence = -1;
      this.rendered = false;
      this.lastVideoTime = -1;
      this.lastProcessAt = 0;
      this.minFrameInterval = 250;
      this.maxNeuralDimension = 640;
      this.maxConcurrentFrames = 1;
      this.neuralEnabled = true;
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
      const extensionApi = globalThis.browser ?? globalThis.chrome;
      this.model = await new WebGPUModelLoader(
        this.device,
        extensionApi.runtime.getURL("webgpu/models/realesr-general-x4v3"),
      ).load();
      const module = this.device.createShaderModule({ code: displayShader });
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
      this.displayTexturePipeline = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: {
          module: this.device.createShaderModule({ code: `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var sampler0: sampler;
@fragment fn fragment(input: VertexOutput) -> @location(0) vec4f {
  return textureSample(image, sampler0, input.uv);
}` }), entryPoint: "fragment", targets: [{ format: this.format }],
        },
        primitive: { topology: "triangle-list" },
      });
      this.convPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code: convShader }), entryPoint: "main" },
      });
      this.convertPipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code: convertShader }), entryPoint: "main" },
      });
      this.shufflePipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code: shuffleShader }), entryPoint: "main" },
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

    uniform(values) {
      const buffer = this.device.createBuffer({
        size: Math.max(16, values.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(buffer, 0, values);
      return buffer;
    }

    async neuralFrame(width, height) {
      this.device.pushErrorScope("validation");
      this.device.pushErrorScope("internal");
      const inputSize = width * height * 3 * 4;
      let input = this.device.createBuffer({ size: inputSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const originalInput = input;
      const transientBuffers = [input];
      const external = this.device.importExternalTexture({ source: this.video });
      const convertParams = this.uniform(new Uint32Array([width, height]));
      transientBuffers.push(convertParams);
      let bind = this.device.createBindGroup({
        layout: this.convertPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: external },
          { binding: 1, resource: input },
          { binding: 2, resource: { buffer: convertParams } },
          { binding: 3, resource: this.sampler },
        ],
      });
      const encoder = this.device.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(this.convertPipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();

      const convs = this.model.weightsFor("convs.", ".weight");
      for (let i = 0; i < convs.length; i++) {
        const weight = this.model.tensor(`convs.${i}.weight`);
        const bias = this.model.tensor(`convs.${i}.bias`);
        const inputChannels = weight.shape[3];
        const outputChannels = weight.shape[0];
        const output = this.device.createBuffer({
          size: width * height * outputChannels * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const params = this.uniform(new Uint32Array([width, height, inputChannels, outputChannels, i === convs.length - 1 ? 0 : 1]));
        transientBuffers.push(output, params);
        const slope = i < convs.length - 1
          ? this.model.tensor(`acts.${i}.weight`).buffer
          : this.zeroSlope;
        if (!slope) this.zeroSlope = this.device.createBuffer({ size: 64 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        bind = this.device.createBindGroup({
          layout: this.convPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: input } },
            { binding: 1, resource: { buffer: output } },
            { binding: 2, resource: { buffer: weight.buffer } },
            { binding: 3, resource: { buffer: bias.buffer } },
            { binding: 4, resource: { buffer: slope || this.zeroSlope } },
            { binding: 5, resource: { buffer: params } },
          ],
        });
        pass = encoder.beginComputePass();
        pass.setPipeline(this.convPipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), outputChannels);
        pass.end();
        input = output;
      }
      const outputTexture = this.device.createTexture({
        size: [width * 4, height * 4],
        format: "rgba16float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      const shuffleParams = this.uniform(new Uint32Array([width, height, 48, 0]));
      bind = this.device.createBindGroup({
        layout: this.shufflePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input } },
          { binding: 1, resource: outputTexture.createView() },
          { binding: 2, resource: { buffer: shuffleParams } },
          { binding: 3, resource: { buffer: originalInput } },
        ],
      });
      transientBuffers.push(shuffleParams);
      pass = encoder.beginComputePass();
      pass.setPipeline(this.shufflePipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(width * 4 / 8), Math.ceil(height * 4 / 8));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      transientBuffers.forEach((buffer) => buffer.destroy());
      const internalError = await this.device.popErrorScope();
      const validationError = await this.device.popErrorScope();
      if (internalError || validationError) {
        throw new Error(`WebGPU neural: ${(internalError || validationError).message}`);
      }
      return outputTexture;
    }

    processingSize(width, height) {
      const longest = Math.max(width, height);
      if (longest <= this.maxNeuralDimension) return [width, height];
      const scale = this.maxNeuralDimension / longest;
      return [
        Math.max(1, Math.floor(width * scale)),
        Math.max(1, Math.floor(height * scale)),
      ];
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
      await this.renderOriginalFrame();
      this.rendered = true;
      this.resolveFirstFrame?.();
      this.frame();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.video);
      window.addEventListener("scroll", this.resize, { passive: true });
      await this.firstFrame;
      return true;
    }

    async renderOriginalFrame() {
      const external = this.device.importExternalTexture({ source: this.video });
      const bind = this.device.createBindGroup({
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
      pass.setBindGroup(0, bind);
      pass.draw(6);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    }

    async processFrame(width, height, sequence) {
      this.inFlight += 1;
      try {
        if (!this.neuralEnabled) {
          await this.renderOriginalFrame();
          if (!this.rendered) {
            this.rendered = true;
            this.resolveFirstFrame?.();
          }
          return;
        }
        const [processWidth, processHeight] = this.processingSize(width, height);
        const outputTexture = await this.neuralFrame(processWidth, processHeight);
        if (!this.running || sequence < this.lastDisplayedSequence) {
          setTimeout(() => outputTexture.destroy(), 1000);
          return;
        }
        const displayBindGroup = this.device.createBindGroup({
          layout: this.displayTexturePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: outputTexture.createView() },
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
        pass.setPipeline(this.displayTexturePipeline);
        pass.setBindGroup(0, displayBindGroup);
        pass.draw(6);
        pass.end();
        this.lastDisplayedSequence = sequence;
        this.device.queue.submit([encoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        setTimeout(() => outputTexture.destroy(), 1000);
        if (!this.rendered) {
          this.rendered = true;
          this.video.style.visibility = "hidden";
          this.resolveFirstFrame?.();
        }
      } catch (error) {
        this.canvas.dataset.error = error instanceof Error ? error.message : String(error);
        this.video.style.visibility = "";
        this.neuralEnabled = false;
        await this.renderOriginalFrame();
        if (!this.rendered) {
          this.rendered = true;
          this.resolveFirstFrame?.();
        }
      } finally {
        this.inFlight -= 1;
      }
    }

    async frame() {
      if (!this.running) return;
      const width = this.video.videoWidth;
      const height = this.video.videoHeight;
      const now = performance.now();
      const videoTime = this.video.currentTime;
      const canProcess = document.visibilityState === "visible"
        && (!this.rendered || !this.video.paused)
        && videoTime !== this.lastVideoTime
        && now - this.lastProcessAt >= this.minFrameInterval;
      if (canProcess && this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && width > 0 && height > 0) {
        if (this.inFlight < this.maxConcurrentFrames) {
          this.lastVideoTime = videoTime;
          this.lastProcessAt = now;
          const sequence = this.frameSequence++;
          void this.processFrame(width, height, sequence);
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
