export type AppConfig = {
  apiBaseUrl: string;
  liveEventsUrl: string;
  liveWebSocketUrl: string;
  demoMode: boolean;
};

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const appConfig: AppConfig = {
  apiBaseUrl,
  liveEventsUrl:
    import.meta.env.VITE_LIVE_EVENTS_URL ?? `${apiBaseUrl}/api/v1/dashboard/events`,
  liveWebSocketUrl:
    import.meta.env.VITE_LIVE_WS_URL ??
    apiBaseUrl.replace(/^http/, "ws") + "/api/v1/dashboard/ws",
  demoMode: import.meta.env.VITE_DEMO_MODE === "true",
};
