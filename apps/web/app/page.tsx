'use client';

import { useChat } from '@ai-sdk/react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

type Mode = 'naive' | 'hybrid';

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: 'naive', label: 'Búsqueda simple', hint: 'vectorial — rápida' },
  { id: 'hybrid', label: 'Búsqueda avanzada', hint: 'grafo de conocimiento' },
];

const SUGERENCIAS = [
  '¿Qué estudios existen sobre la eficiencia del gasto en CTI?',
  '¿Qué dice la Encuesta Nacional de Percepción 2024?',
  '¿Qué indicadores bibliométricos reporta el CONCYTEC para 2018–2024?',
];

export default function Chat() {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<Mode>('naive');
  const { messages, sendMessage, status, error } = useChat();
  const busy = status === 'submitted' || status === 'streaming';

  function ask(text: string) {
    if (!text.trim() || busy) return;
    sendMessage({ text }, { body: { mode } });
    setInput('');
  }

  return (
    <div className="mx-auto flex h-dvh max-w-3xl flex-col px-4">
      <header className="border-b border-zinc-200 py-4 dark:border-zinc-800">
        <h1 className="text-lg font-semibold">
          Publicaciones y eventos institucionales · CONCYTEC
        </h1>
        <p className="text-sm text-zinc-500">
          Chatbot sobre los documentos del repositorio institucional. Las
          respuestas citan documento, enlace y página.
        </p>
        <div className="mt-3 flex flex-wrap gap-2" role="radiogroup" aria-label="Modo de búsqueda">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="radio"
              aria-checked={mode === m.id}
              onClick={() => setMode(m.id)}
              className={`rounded-full border px-3 py-1 text-sm transition-colors ${
                mode === m.id
                  ? 'border-emerald-600 bg-emerald-600 text-white'
                  : 'border-zinc-300 text-zinc-600 hover:border-zinc-500 dark:border-zinc-700 dark:text-zinc-300'
              }`}
            >
              {m.label} <span className="opacity-70">({m.hint})</span>
            </button>
          ))}
        </div>
      </header>

      <main className="flex-1 space-y-4 overflow-y-auto py-4">
        {messages.length === 0 && (
          <div className="mt-8 space-y-2 text-center text-sm text-zinc-500">
            <p>Prueba con una de estas preguntas:</p>
            {SUGERENCIAS.map((s) => (
              <button
                key={s}
                onClick={() => ask(s)}
                className="block w-full rounded-lg border border-zinc-200 px-3 py-2 text-left hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900"
              >
                {s}
              </button>
            ))}
          </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
            className={
              message.role === 'user'
                ? 'ml-auto w-fit max-w-[85%] rounded-2xl bg-emerald-600 px-4 py-2 text-white'
                : 'max-w-[95%] rounded-2xl bg-zinc-100 px-4 py-3 dark:bg-zinc-900'
            }
          >
            {message.parts.map((part, i) =>
              part.type === 'text' ? (
                message.role === 'user' ? (
                  <p key={i} className="whitespace-pre-wrap">{part.text}</p>
                ) : (
                  <div
                    key={i}
                    className="space-y-2 text-sm leading-relaxed [&_a]:text-emerald-700 [&_a]:underline dark:[&_a]:text-emerald-400 [&_li]:ml-5 [&_ol]:list-decimal [&_strong]:font-semibold [&_ul]:list-disc"
                  >
                    <ReactMarkdown>{part.text}</ReactMarkdown>
                  </div>
                )
              ) : null,
            )}
          </div>
        ))}
        {status === 'submitted' && (
          <p className="text-sm text-zinc-500">Buscando en los documentos…</p>
        )}
        {error && (
          <p className="text-sm text-red-600">
            Ocurrió un error al consultar. Intenta de nuevo.
          </p>
        )}
      </main>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="flex gap-2 border-t border-zinc-200 py-4 dark:border-zinc-800"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.currentTarget.value)}
          placeholder="Escribe tu pregunta sobre las publicaciones del CONCYTEC…"
          className="flex-1 rounded-xl border border-zinc-300 px-4 py-2 outline-none focus:border-emerald-600 dark:border-zinc-700 dark:bg-zinc-900"
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded-xl bg-emerald-600 px-4 py-2 font-medium text-white disabled:opacity-40"
        >
          Enviar
        </button>
      </form>
    </div>
  );
}
