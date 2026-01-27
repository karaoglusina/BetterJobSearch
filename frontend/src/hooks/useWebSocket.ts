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

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/chat`);

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);

    ws.onmessage = (event) => {
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
          { role: 'system', content: `Error: ${data.message}` },
        ]);
        setIsLoading(false);
      }
    };

    wsRef.current = ws;
    return () => ws.close();
  }, []);

  const sendMessage = useCallback((content: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    setMessages((prev) => [...prev, { role: 'user', content }]);
    setIsLoading(true);
    wsRef.current.send(JSON.stringify({ type: 'message', content }));
  }, []);

  const resetChat = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: 'reset' }));
    setMessages([]);
  }, []);

  return { messages, sendMessage, resetChat, isConnected, isLoading };
}
