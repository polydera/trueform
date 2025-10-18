export default function numKM(value: number): string {
  if (value >= 1_000_000) {
    return `${+(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${+(value / 1_000).toFixed(1)}k`;
  }
  return +value.toFixed(1) + '';
}
