const API_BASE = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail =
      typeof payload === "object" && payload !== null
        ? payload.detail || payload.message
        : payload;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload;
}

export function dateQuery(startDate, endDate) {
  if (!startDate || !endDate) return "";
  const params = new URLSearchParams({
    start_date: startDate,
    end_date: endDate,
  });
  return `?${params}`;
}
