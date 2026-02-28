let _tf: any = null;
let _promise: Promise<any> | null = null;

export function useTrueform() {
  const load = async () => {
    if (_tf) return _tf;
    if (!_promise) {
      _promise = import("@/examples/trueform/index.js");
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
