import { appConfig } from "../config";
import type { DashboardEvent } from "../types/dashboard";

export type LiveConnection = {
  close: () => void;
};

export function connectLiveUpdates(
  onEvent: (event: DashboardEvent) => void,
  onStatus: (status: "connected" | "connecting" | "disconnected") => void,
): LiveConnection {
  if ("EventSource" in window) {
    const source = new EventSource(appConfig.liveEventsUrl);
    onStatus("connecting");
    source.onopen = () => onStatus("connected");
    source.onerror = () => onStatus("disconnected");
    source.onmessage = (message) => {
      onEvent(JSON.parse(message.data) as DashboardEvent);
    };
    return { close: () => source.close() };
  }

  const socket = new WebSocket(appConfig.liveWebSocketUrl);
  onStatus("connecting");
  socket.onopen = () => onStatus("connected");
  socket.onclose = () => onStatus("disconnected");
  socket.onerror = () => onStatus("disconnected");
  socket.onmessage = (message) => {
    onEvent(JSON.parse(String(message.data)) as DashboardEvent);
  };
  return { close: () => socket.close() };
}
