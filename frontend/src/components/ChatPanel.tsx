import React, { useRef, useEffect, useState } from 'react';
import { useWebSocket, ChatMessage } from '../hooks/useWebSocket';

interface ChatPanelProps {
  onJobHighlight?: (jobIds: string[]) => void;
}

export default function ChatPanel({ onJobHighlight }: ChatPanelProps) {
  const { messages, sendMessage, resetChat, isConnected, isLoading } = useWebSocket();
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessage(input.trim());
    setInput('');
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100%',
      background: 'var(--bg-secondary)', borderLeft: '1px solid var(--border)',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px', borderBottom: '1px solid var(--border)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{ fontWeight: 600, fontSize: 14 }}>Chat</span>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{
            width: 8, height: 8, borderRadius: '50%',
            background: isConnected ? 'var(--success)' : 'var(--error)',
          }} />
          <button onClick={resetChat} style={{
            background: 'transparent', border: '1px solid var(--border)',
            color: 'var(--text-secondary)', padding: '4px 8px', borderRadius: 4,
            cursor: 'pointer', fontSize: 12,
          }}>Reset</button>
        </div>
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
        {messages.length === 0 && (
          <div style={{ color: 'var(--text-secondary)', fontSize: 13, textAlign: 'center', marginTop: 40 }}>
            Ask about jobs, compare positions, or explore the market.
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {isLoading && (
          <div style={{ color: 'var(--text-secondary)', fontSize: 13, padding: '8px 0' }}>
            Thinking...
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} style={{
        padding: 12, borderTop: '1px solid var(--border)',
        display: 'flex', gap: 8,
      }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isConnected ? 'Ask about jobs...' : 'Connecting...'}
          disabled={!isConnected || isLoading}
          style={{
            flex: 1, padding: '8px 12px', borderRadius: 6,
            border: '1px solid var(--border)', background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)', fontSize: 13, outline: 'none',
          }}
        />
        <button type="submit" disabled={!isConnected || isLoading || !input.trim()} style={{
          padding: '8px 16px', borderRadius: 6, border: 'none',
          background: 'var(--accent)', color: 'white', cursor: 'pointer',
          fontSize: 13, fontWeight: 500,
          opacity: (!isConnected || isLoading || !input.trim()) ? 0.5 : 1,
        }}>
          Send
        </button>
      </form>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  return (
    <div style={{
      marginBottom: 12,
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
    }}>
      <div style={{
        maxWidth: '85%', padding: '8px 12px', borderRadius: 8,
        background: isUser ? 'var(--accent)' : 'var(--bg-tertiary)',
        color: 'var(--text-primary)', fontSize: 13, lineHeight: 1.5,
        whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      }}>
        {message.content}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div style={{
            marginTop: 8, paddingTop: 8,
            borderTop: '1px solid rgba(255,255,255,0.1)',
            fontSize: 11, color: 'var(--text-secondary)',
          }}>
            Tools used: {message.toolCalls.map((tc) => tc.tool).join(', ')}
          </div>
        )}
      </div>
    </div>
  );
}
