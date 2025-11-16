import { geolocation } from "@vercel/functions";
import {
  convertToModelMessages,
  createUIMessageStream,
  JsonToSseTransformStream,
  smoothStream,
  streamText,
} from "ai";
import { unstable_cache as cache } from "next/cache";
import { after } from "next/server";
import {
  createResumableStreamContext,
  type ResumableStreamContext,
} from "resumable-stream";
import type { ModelCatalog } from "tokenlens/core";
import { fetchModels } from "tokenlens/fetch";
import { getUsage } from "tokenlens/helpers";
import { auth, type UserType } from "@/app/(auth)/auth";
import type { VisibilityType } from "@/components/visibility-selector";
import { entitlementsByUserType } from "@/lib/ai/entitlements";
import { generateEmbedding } from "@/lib/ai/embeddings";
import type { ChatModel } from "@/lib/ai/models";
import { myProvider } from "@/lib/ai/providers";
import { isProductionEnvironment } from "@/lib/constants";
import {
  createStreamId,
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
  searchPerfumesByEmbedding,
  updateChatLastContextById,
} from "@/lib/db/queries";
import type { DBMessage } from "@/lib/db/schema";
import { ChatSDKError } from "@/lib/errors";
import type { ChatMessage } from "@/lib/types";
import type { AppUsage } from "@/lib/usage";
import { convertToUIMessages, generateUUID, getTextFromMessage } from "@/lib/utils";
import { generateTitleFromUserMessage } from "../../actions";
import { type PostRequestBody, postRequestBodySchema } from "./schema";

export const maxDuration = 60;

let globalStreamContext: ResumableStreamContext | null = null;

const getTokenlensCatalog = cache(
  async (): Promise<ModelCatalog | undefined> => {
    try {
      return await fetchModels();
    } catch (err) {
      console.warn(
        "TokenLens: catalog fetch failed, using default catalog",
        err
      );
      return; // tokenlens helpers will fall back to defaultCatalog
    }
  },
  ["tokenlens-catalog"],
  { revalidate: 24 * 60 * 60 } // 24 hours
);

export function getStreamContext() {
  if (!globalStreamContext) {
    try {
      globalStreamContext = createResumableStreamContext({
        waitUntil: after,
      });
    } catch (error: any) {
      if (error.message.includes("REDIS_URL")) {
        console.log(
          " > Resumable streams are disabled due to missing REDIS_URL"
        );
      } else {
        console.error(error);
      }
    }
  }

  return globalStreamContext;
}

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (_) {
    return new ChatSDKError("bad_request:api").toResponse();
  }

  try {
    const {
      id,
      message,
      selectedChatModel,
      selectedVisibilityType,
    }: {
      id: string;
      message: ChatMessage;
      selectedChatModel: ChatModel["id"];
      selectedVisibilityType: VisibilityType;
    } = requestBody;

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError("unauthorized:chat").toResponse();
    }

    const userType: UserType = session.user.type;

    const messageCount = await getMessageCountByUserId({
      id: session.user.id,
      differenceInHours: 24,
    });

    if (messageCount > entitlementsByUserType[userType].maxMessagesPerDay) {
      return new ChatSDKError("rate_limit:chat").toResponse();
    }

    const chat = await getChatById({ id });
    let messagesFromDb: DBMessage[] = [];

    if (chat) {
      if (chat.userId !== session.user.id) {
        return new ChatSDKError("forbidden:chat").toResponse();
      }
      // Only fetch messages if chat already exists
      messagesFromDb = await getMessagesByChatId({ id });
    } else {
      const title = await generateTitleFromUserMessage({
        message,
      });

      await saveChat({
        id,
        userId: session.user.id,
        title,
        visibility: selectedVisibilityType,
      });
      // New chat - no need to fetch messages, it's empty
    }

    const uiMessages = [...convertToUIMessages(messagesFromDb), message];

    await saveMessages({
      messages: [
        {
          chatId: id,
          id: message.id,
          role: "user",
          parts: message.parts,
          attachments: [],
          createdAt: new Date(),
        },
      ],
    });

    const streamId = generateUUID();
    await createStreamId({ streamId, chatId: id });

    let finalMergedUsage: AppUsage | undefined;

    // Wyciągnij tekst z wiadomości użytkownika
    const userMessageText = getTextFromMessage(message);

    const stream = createUIMessageStream({
      execute: async ({ writer: dataStream }) => {
        try {
          // Generuj embedding dla zapytania użytkownika
          const queryEmbedding = await generateEmbedding(userMessageText);

          // Wyszukaj perfumy używając vector search
          const perfumes = await searchPerfumesByEmbedding({
            queryEmbedding,
            limit: 5,
          });

          // Przygotuj kontekst z znalezionymi perfumami
          const perfumesContext = perfumes.length > 0
            ? perfumes
                .map(
                  (p) =>
                    `**${p.perfume_name}**${p.brand ? ` (${p.brand})` : ""}\n` +
                    `${p.description ? `Opis: ${p.description}\n` : ""}` +
                    `${p.rating ? `⭐ Ocena: ${p.rating}/5${p.rating_count ? ` (${p.rating_count} opinii)` : ""}\n` : ""}` +
                    `${p.notes ? `Nuty zapachowe: ${Array.isArray(p.notes) ? p.notes.join(", ") : JSON.stringify(p.notes)}\n` : ""}` +
                    `${p.season ? `Sezon: ${Array.isArray(p.season) ? p.season.join(", ") : JSON.stringify(p.season)}\n` : ""}` +
                    `${p.gender ? `Dla: ${Array.isArray(p.gender) ? p.gender.join(", ") : JSON.stringify(p.gender)}\n` : ""}` +
                    `${p.longevity ? `Trwałość: ${JSON.stringify(p.longevity)}\n` : ""}` +
                    `${p.sillage ? `Sillage: ${JSON.stringify(p.sillage)}\n` : ""}` +
                    `${p.timeOfDay ? `Pora dnia: ${Array.isArray(p.timeOfDay) ? p.timeOfDay.join(", ") : JSON.stringify(p.timeOfDay)}\n` : ""}` +
                    `${p.valueForMoney ? `Stosunek jakości do ceny: ${JSON.stringify(p.valueForMoney)}\n` : ""}` +
                    `${p.pros ? `Zalety: ${Array.isArray(p.pros) ? p.pros.join(", ") : JSON.stringify(p.pros)}\n` : ""}` +
                    `${p.cons ? `Wady: ${Array.isArray(p.cons) ? p.cons.join(", ") : JSON.stringify(p.cons)}\n` : ""}` +
                    `${p.similarPerfumes ? `Podobne perfumy: ${Array.isArray(p.similarPerfumes) ? p.similarPerfumes.join(", ") : JSON.stringify(p.similarPerfumes)}\n` : ""}`
                )
                .join("\n\n---\n\n")
            : "";

          // Generuj odpowiedź używając AI modelu do formatowania
          const systemPrompt = `Jesteś ekspertem w dziedzinie perfum - profesjonalnym asystentem perfumowym z głęboką wiedzą o zapachach, nutach zapachowych, kompozycjach i trendach w świecie perfum.

Twoje zadania:
1. Pomagasz użytkownikom znaleźć idealne perfumy na podstawie ich preferencji, okazji, budżetu i stylu życia
2. Wyjaśniasz charakterystykę zapachów, nuty zapachowe i kompozycje
3. Rekomendujesz perfumy na podstawie podobieństwa, okazji, sezonu i innych kryteriów
4. Porównujesz perfumy i pomagasz w wyborze
5. Udzielasz profesjonalnych porad dotyczących aplikacji, przechowywania i pielęgnacji perfum

Styl komunikacji:
- Bądź profesjonalny, ale przyjazny i przystępny
- Używaj terminologii perfumeryjnej, ale wyjaśniaj trudne pojęcia
- Bądź entuzjastyczny, ale obiektywny
- Zawsze odpowiadaj po polsku
- Formatuj odpowiedzi w sposób czytelny, używając akapitów i list gdy to pomocne

${perfumes.length > 0 ? `Znalezione perfumy w bazie danych:\n\n${perfumesContext}\n\nUżyj tych informacji, aby udzielić szczegółowej i pomocnej odpowiedzi. Jeśli użytkownik pyta o konkretne perfumy, skup się na tych znalezionych w bazie. Jeśli pyta ogólnie, możesz użyć znalezionych perfum jako przykładów lub punktu wyjścia.` : "Nie znaleziono perfum pasujących do zapytania w bazie danych. Odpowiedz pomocnie, sugerując inne podejście do wyszukiwania lub zadaj pytania pomocnicze, aby lepiej zrozumieć potrzeby użytkownika."}

Pamiętaj: Zawsze odpowiadaj po polsku i bądź pomocny, profesjonalny i entuzjastyczny w temacie perfum.`;

          const result = streamText({
            model: myProvider.languageModel(selectedChatModel),
            system: systemPrompt,
            messages: [
              {
                role: "user",
                content: userMessageText,
              },
            ],
            experimental_transform: smoothStream({ chunking: "word" }),
            experimental_telemetry: {
              isEnabled: isProductionEnvironment,
              functionId: "stream-text",
            },
            onFinish: async ({ usage }) => {
              try {
                const providers = await getTokenlensCatalog();
                const modelId =
                  myProvider.languageModel(selectedChatModel).modelId;
                if (!modelId) {
                  finalMergedUsage = usage;
                  dataStream.write({
                    type: "data-usage",
                    data: finalMergedUsage,
                  });
                  return;
                }

                if (!providers) {
                  finalMergedUsage = usage;
                  dataStream.write({
                    type: "data-usage",
                    data: finalMergedUsage,
                  });
                  return;
                }

                const summary = getUsage({ modelId, usage, providers });
                finalMergedUsage = { ...usage, ...summary, modelId } as AppUsage;
                dataStream.write({ type: "data-usage", data: finalMergedUsage });
              } catch (err) {
                console.warn("TokenLens enrichment failed", err);
                finalMergedUsage = usage;
                dataStream.write({ type: "data-usage", data: finalMergedUsage });
              }
            },
          });

          result.consumeStream();

          dataStream.merge(
            result.toUIMessageStream({
              sendReasoning: false,
            })
          );
        } catch (error) {
          console.error("Error in perfume search:", error);
          dataStream.write({
            type: "text-delta",
            delta: "Przepraszam, wystąpił błąd podczas wyszukiwania perfum. Spróbuj ponownie.",
            id: generateUUID(),
          });
          dataStream.write({ type: "finish" });
        }
      },
      generateId: generateUUID,
      onFinish: async ({ messages }) => {
        await saveMessages({
          messages: messages.map((currentMessage) => ({
            id: currentMessage.id,
            role: currentMessage.role,
            parts: currentMessage.parts,
            createdAt: new Date(),
            attachments: [],
            chatId: id,
          })),
        });

        if (finalMergedUsage) {
          try {
            await updateChatLastContextById({
              chatId: id,
              context: finalMergedUsage,
            });
          } catch (err) {
            console.warn("Unable to persist last usage for chat", id, err);
          }
        }
      },
      onError: () => {
        return "Oops, an error occurred!";
      },
    });

    // const streamContext = getStreamContext();

    // if (streamContext) {
    //   return new Response(
    //     await streamContext.resumableStream(streamId, () =>
    //       stream.pipeThrough(new JsonToSseTransformStream())
    //     )
    //   );
    // }

    return new Response(stream.pipeThrough(new JsonToSseTransformStream()));
  } catch (error) {
    const vercelId = request.headers.get("x-vercel-id");

    if (error instanceof ChatSDKError) {
      return error.toResponse();
    }

    console.error("Unhandled error in chat API:", error, { vercelId });
    return new ChatSDKError("offline:chat").toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new ChatSDKError("bad_request:api").toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError("unauthorized:chat").toResponse();
  }

  const chat = await getChatById({ id });

  if (chat?.userId !== session.user.id) {
    return new ChatSDKError("forbidden:chat").toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
