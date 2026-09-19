class WebGPUModelLoader {
  constructor(device, baseUrl) {
    this.device = device;
    this.baseUrl = baseUrl;
  }

  async load() {
    const manifestResponse = await fetch(`${this.baseUrl}/manifest.json`);
    if (!manifestResponse.ok) throw new Error(`Modelo: HTTP ${manifestResponse.status}`);
    this.manifest = await manifestResponse.json();
    if (this.manifest.format !== "video-upscaler-webgpu") {
      throw new Error("Formato de modelo WebGPU desconhecido");
    }
    const weightsResponse = await fetch(`${this.baseUrl}/weights.bin`);
    if (!weightsResponse.ok) throw new Error(`Pesos: HTTP ${weightsResponse.status}`);
    this.weights = await weightsResponse.arrayBuffer();
    this.buffers = new Map();
    for (const [name, entry] of Object.entries(this.manifest.weights)) {
      const bytes = this.weights.slice(entry.offset * 4, (entry.offset + entry.length) * 4);
      const buffer = this.device.createBuffer({
        size: Math.max(4, (bytes.byteLength + 3) & ~3),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(buffer, 0, bytes);
      this.buffers.set(name, buffer);
    }
    return this;
  }

  tensor(name) {
    const tensor = this.manifest.weights[name];
    if (!tensor) throw new Error(`Tensor ausente: ${name}`);
    return { buffer: this.buffers.get(name), shape: tensor.shape, layout: tensor.layout };
  }

  weightsFor(prefix, suffix = "") {
    const names = Object.keys(this.manifest.weights)
      .filter((name) => name.startsWith(prefix) && name.endsWith(suffix))
      .sort((a, b) => {
        const ai = Number(a.split(".")[1]);
        const bi = Number(b.split(".")[1]);
        return ai - bi;
      });
    return names.map((name) => this.tensor(name));
  }
}

globalThis.WebGPUModelLoader = WebGPUModelLoader;
