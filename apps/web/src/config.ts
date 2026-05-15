export type AppConfig = {
  apiBaseUrl: string;
};

export const appConfig: AppConfig = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
};

