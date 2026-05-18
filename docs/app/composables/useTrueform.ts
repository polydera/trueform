let _tf: any = null;
let _promise: Promise<any> | null = null;

export function useTrueform() {
  const load = async () => {
    if (_tf) return _tf;
    if (!_promise) {
      _promise = (async () => {
        if (import.meta.env.DEV) {
          return await import("@polydera/trueform");
        }
        const tf = await import("@polydera/trueform/manual");
        const { trueformVersion } = useRuntimeConfig().public;
        const base = `https://cdn.jsdelivr.net/npm/@polydera/trueform@${trueformVersion}/dist`;
        await tf.init({
          wasmUrl: await tf.toBlobURL(`${base}/trueform_wasm.wasm`, "application/wasm"),
          workerUrl: await tf.toBlobURL(`${base}/trueform_wasm.js`, "text/javascript"),
        });
        return tf;
      })();
    }
    try {
      _tf = await _promise;
      return _tf;
    } catch (error) {
      _promise = null;
      throw error;
    }
  };
  return { load };
}
