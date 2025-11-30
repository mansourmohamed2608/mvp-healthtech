import { useEffect, useRef } from 'react';

export interface TranscriptItem {
  role: 'user' | 'assistant' | 'partial';
  text: string;
  ts: number;
}

export function StreamingTranscript({ items }: { items: TranscriptItem[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView({ behavior: 'smooth' });
  }, [items]);

  return (
    <div className="bg-white/70 dark:bg-slate-800/70 rounded-xl p-4 shadow-sm border border-slate-200 dark:border-slate-700 h-72 overflow-y-auto">
      {items.map((item, idx) => (
        <div key={idx} className="mb-2 text-sm">
          <span className="font-semibold mr-2">
            {item.role === 'assistant' ? 'Assistant' : item.role === 'partial' ? '…' : 'User'}:
          </span>
          <span className={item.role === 'partial' ? 'italic text-slate-500' : ''}>{item.text}</span>
        </div>
      ))}
      <div ref={ref} />
    </div>
  );
}
