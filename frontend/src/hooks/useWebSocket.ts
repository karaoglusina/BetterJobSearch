import { useCallback, useEffect, useRef, useState } from 'react';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  toolCalls?: { tool: string; args: Record<string, unknown> }[];
  tokens?: number;
}

export interface WsMessage {
  type: 'answer' | 'tool_call' | 'intent' | 'ui_action' | 'error' | 'reset_ack';
  content?: string;
  tool?: string;
  args?: Record<string, unknown>;
  result?: string;
  error?: string;
  intent?: string;
  confidence?: number;
  job_ids?: string[];
  tool_calls?: { tool: string; args: Record<string, unknown> }[];
  model?: string;
  total_tokens?: number;
  action?: string;
  data?: unknown;
  message?: string;
}

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_BASE_DELAY = 1000;

export interface WebSocketContext {
  selectedJobIds?: string[];
  currentAspect?: string;
  currentFilters?: Record<string, string>;
}

export function useWebSocket(
  onSetJobs?: (jobIds: string[]) => void,
  context?: WebSocketContext
) {
  const wsRef = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const onSetJobsRef = useRef(onSetJobs);
  const contextRef = useRef(context);
  onSetJobsRef.current = onSetJobs;
  contextRef.current = context;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || !mountedRef.current) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/chat`;

    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return; }
      setIsConnected(true);
      reconnectAttempts.current = 0;
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setIsConnected(false);
      setIsLoading(false);

      if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = RECONNECT_BASE_DELAY * Math.pow(2, reconnectAttempts.current);
        reconnectAttempts.current += 1;
        reconnectTimer.current = setTimeout(() => {
          if (mountedRef.current) connect();
        }, delay);
      }
    };

    ws.onerror = () => {
      if (!mountedRef.current) return;
      setIsConnected(false);
    };

    ws.onmessage = (event) => {
      if (!mountedRef.current) return;
      const data: WsMessage = JSON.parse(event.data);

      if (data.type === 'answer') {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: data.content || '',
            toolCalls: data.tool_calls,
            tokens: data.total_tokens,
          },
        ]);
        setIsLoading(false);
      } else if (data.type === 'error') {
        setMessages((prev) => [
          ...prev,
          { role: 'system', content: `Error: ${data.message || data.error || 'Unknown error'}` },
        ]);
        setIsLoading(false);
      } else if (data.type === 'tool_call') {
        setMessages((prev) => [
          ...prev,
          { role: 'system', content: `Using tool: ${data.tool}` },
        ]);
      } else if (data.type === 'ui_action' && data.action === 'set_jobs' && data.job_ids) {
        if (onSetJobsRef.current) {
          onSetJobsRef.current(data.job_ids);
        }
      }
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((content: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setMessages((prev) => [...prev, { role: 'user', content }]);
    setIsLoading(true);

    // Include context (selected jobs, etc.) in the message
    const payload: Record<string, unknown> = {
      type: 'message',
      content,
    };

    if (contextRef.current) {
      payload.context = contextRef.current;
    }

    wsRef.current.send(JSON.stringify(payload));
  }, []);

  const resetChat = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: 'reset' }));
    setMessages([]);
  }, []);

  return { messages, sendMessage, resetChat, isConnected, isLoading };
}
