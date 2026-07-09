import { streamText, convertToModelMessages, type UIMessage } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';

const google = createGoogleGenerativeAI({ apiKey: process.env.GEMINI_API_KEY });
const RAG_API = process.env.RAG_API_URL ?? 'http://localhost:8000';
const CHAT_MODEL = process.env.CHAT_MODEL ?? 'gemini-3.1-flash-lite';

function systemPrompt(context: string): string {
  return `Eres el asistente del repositorio institucional del CONCYTEC (Consejo Nacional de Ciencia, Tecnología e Innovación Tecnológica del Perú), especializado en la colección "Publicaciones y eventos institucionales".

REGLAS ESTRICTAS:
1. Responde SIEMPRE en español.
2. Responde ÚNICAMENTE con información presente en el CONTEXTO de abajo. No uses conocimiento externo ni inventes datos, cifras, citas o números de página.
3. Si el contexto no contiene la respuesta, responde exactamente: "No encuentro esa información en los documentos del repositorio." y sugiere reformular la pregunta.
4. Solo respondes preguntas sobre las publicaciones y eventos institucionales del CONCYTEC (ciencia, tecnología e innovación en el Perú). Si la pregunta es de otro tema (programación, conocimiento general, etc.), decláralo amablemente y no la respondas.
5. CITAS: los fragmentos del contexto llevan marcadores con el formato [Doc: <título> | Handle: <url> | Página <n>] (a veces con "| Archivo: <nombre>"). Al final de tu respuesta agrega una sección "**Fuentes:**" listando cada documento citado así: "- [<título>](<url>), página <n>". Cita solo documentos y páginas que realmente aparezcan en los marcadores del contexto usado.

CONTEXTO:
${context}`;
}

export async function POST(req: Request) {
  const {
    messages,
    mode = 'naive',
  }: { messages: UIMessage[]; mode?: 'naive' | 'hybrid' } = await req.json();

  const lastUser = [...messages].reverse().find((m) => m.role === 'user');
  const question =
    lastUser?.parts
      .filter((p) => p.type === 'text')
      .map((p) => p.text)
      .join('\n') ?? '';

  const res = await fetch(`${RAG_API}/query`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ question, mode }),
  });
  if (!res.ok) {
    return Response.json(
      { error: 'El servicio de búsqueda no está disponible. Intenta de nuevo.' },
      { status: 502 },
    );
  }
  const { context } = (await res.json()) as { context: string };

  const result = streamText({
    model: google(CHAT_MODEL),
    system: systemPrompt(context),
    messages: await convertToModelMessages(messages),
  });

  return result.toUIMessageStreamResponse();
}
