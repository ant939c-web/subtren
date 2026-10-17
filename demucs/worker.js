// Separa voz y pista con Demucs (htdemucs, 4 fuentes) usando ONNX Runtime Web.
// Usa la tarjeta gráfica (WebGPU) si existe; si no, el procesador (más lento).
import * as ort from "../ort/ort.webgpu.bundle.min.mjs";
import { separateTracks } from "./apply.js";

ort.env.wasm.wasmPaths = new URL("../ort/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;

const RATE = 44100;

class HTDemucs {
  sources = ["drums", "bass", "other", "vocals"];
  audioChannels = 2;
  samplerate = RATE;
  segment = 7.8;
  validLength(length) {
    const training = Math.floor(this.segment * this.samplerate);
    if (training < length) throw new Error(`Pedazo de ${length} muestras más largo que ${training}`);
    return training;
  }
  async forward(mix, magspec) {
    const feeds = {
      [this.session.inputNames[0]]: new ort.Tensor("float32", mix.data, mix.shape),
      [this.session.inputNames[1]]: new ort.Tensor("float32", magspec.data, magspec.shape),
    };
    const r = await this.session.run(feeds);
    const x = r[this.session.outputNames[0]], xt = r[this.session.outputNames[1]];
    return { outX: { data: x.data, shape: x.dims }, outXt: { data: xt.data, shape: xt.dims } };
  }
}

const post = (m, t) => self.postMessage(m, t || []);

async function downloadModel() {
  const base = new URL("../modelo/", import.meta.url);
  const man = await (await fetch(new URL("manifest.json", base))).json();
  let cache = null;
  try { cache = await caches.open("subtren-" + man.version); } catch { cache = null; }
  const all = new Uint8Array(man.total);
  let got = 0;
  for (const part of man.parts) {
    const url = new URL(part.file, base).href;
    let res = null;
    try { res = cache ? await cache.match(url) : null; } catch { res = null; }
    let bytes;
    if (res) {
      bytes = new Uint8Array(await res.arrayBuffer());
      if (bytes.length !== part.size) { res = null; try { await cache.delete(url); } catch {} }
    }
    if (!res) {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`No se pudo bajar la inteligencia artificial (error ${r.status})`);
      bytes = new Uint8Array(part.size);
      const reader = r.body.getReader();
      let o = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (o + value.length > part.size) throw new Error("La descarga llegó dañada");
        bytes.set(value, o);
        o += value.length;
        post({ type: "progress", phase: "download", p: (got + o) / man.total, total: man.total });
      }
      if (o !== part.size) throw new Error("La descarga se cortó. Revisa el internet y vuelve a intentar.");
      if (cache) { try { await cache.put(url, new Response(bytes)); } catch {} }
    }
    all.set(bytes, got);
    got += part.size;
    post({ type: "progress", phase: "download", p: got / man.total, total: man.total });
  }
  return all;
}

async function loadModel() {
  const bytes = await downloadModel();
  post({ type: "progress", phase: "load", p: 0 });
  const model = new HTDemucs();
  let device = "cpu";
  let gpu = null;
  try { gpu = navigator.gpu ? await navigator.gpu.requestAdapter() : null; } catch { gpu = null; }
  if (gpu) {
    try {
      model.session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
      device = "gpu";
    } catch (e) {
      console.warn("WebGPU no funcionó, uso el procesador", e);
    }
  }
  if (!model.session) model.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return { model, device };
}

let modelP = null;

self.onmessage = async (e) => {
  const { L, R } = e.data;
  try {
    if (!modelP) modelP = loadModel();
    let ready;
    try { ready = await modelP; } catch (err) { modelP = null; throw err; }
    const { model, device } = ready;
    post({ type: "progress", phase: "separate", p: 0, device });
    // Se procesa en ventanas de 40 s con 3 s de contexto a cada lado para no llenar la memoria.
    const n = L.length, WIN = RATE * 40, M = RATE * 3;
    const vL = new Float32Array(n), vR = new Float32Array(n), iL = new Float32Array(n), iR = new Float32Array(n);
    const starts = [];
    for (let s = 0; s < n; s += WIN) starts.push(s);
    for (let w = 0; w < starts.length; w++) {
      const s = starts[w], end = Math.min(n, s + WIN), a = Math.max(0, s - M), b = Math.min(n, end + M);
      const tracks = await separateTracks(
        model,
        { channelData: [L.subarray(a, b), R.subarray(a, b)], sampleRate: RATE },
        (i, total) => post({ type: "progress", phase: "separate", p: (w + i / total) / starts.length, device }),
        0.25,
      );
      for (let c = 0; c < 2; c++) {
        const v = tracks.vocals.channelData[c], d = tracks.drums.channelData[c];
        const ba = tracks.bass.channelData[c], ot = tracks.other.channelData[c];
        const V = c ? vR : vL, I = c ? iR : iL;
        for (let i = s; i < end; i++) { const k = i - a; V[i] = v[k]; I[i] = d[k] + ba[k] + ot[k]; }
      }
    }
    post({ type: "done", vL, vR, iL, iR, device }, [vL.buffer, vR.buffer, iL.buffer, iR.buffer]);
  } catch (err) {
    post({ type: "error", message: String((err && err.message) || err) });
  }
};
