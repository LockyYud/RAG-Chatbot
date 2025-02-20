"use client";

import { useLocalStorage } from "@/libs/hooks/use-local-storage";
import { useEffect, useRef, useState } from "react";
import { ChatList } from "./chat-list";
import { ChatState, Message } from "@/libs/types";
import { ChatPanel } from "./chat-panel";
import { useScrollAnchor } from "@/libs/hooks/use-scroll-anchor";
import { cn } from "@/libs/utils";
import { EmptyScreen } from "./empty-screen";
import { useChatContext } from "@/libs/context/chat-context";
import { message } from "antd";
import OpenAI from "openai";
import tools from "./const";

export function Chat() {
    const [apiKey, setApiKey] = useState<string>("");
    const openai = new OpenAI({
        apiKey: apiKey ?? "",
        dangerouslyAllowBrowser: true,
    });
    useEffect(() => {
        const savedApiKey = localStorage.getItem("OPENAI_API_KEY") || "";
        setApiKey(savedApiKey);
    }, []);
    const { setState, id: conversationId } = useChatContext();
    const [listMessage, setListMessage] = useState<Message[]>([]);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_, setNewChatId] = useLocalStorage("newChatId", conversationId);
    const chatPanelRef = useRef<HTMLDivElement>(null);
    const [height, setHeight] = useState<number>(0);
    const listRef = useRef<HTMLDivElement>(null);
    const [correctMessage, setCorrectMessage] = useState<string>("");
    const [isCorrectMessage, setIsCorrectMessage] = useState<boolean>();
    useEffect(() => {
        if (listRef.current) {
            listRef.current.scrollTop = window.innerHeight;
        }
        window.scrollTo(0, document.body.scrollHeight);
    }, [listMessage]);
    useEffect(() => {
        if (chatPanelRef.current) {
            const resizeObserver = new ResizeObserver((entries) => {
                for (const entry of entries) {
                    setHeight(entry.contentRect.height); // Get the height dynamically
                }
            });

            resizeObserver.observe(chatPanelRef.current);

            // Cleanup observer on component unmount
            return () => {
                resizeObserver.disconnect();
            };
        }
    }, []);
    useEffect(() => {
        if (
            listMessage.at(-1) &&
            listMessage.at(-1)?.role == "user" &&
            localStorage.getItem("api_base_url")
        ) {
            try {
                openai.chat.completions
                    .create({
                        model: "gpt-4o-mini",
                        messages: [
                            {
                                role: "system",
                                content:
                                    "You are an interviewer, and we will conduct an interview for a Data Scientist position. The interview will cover past experiences, theoretical discussions, math/statistics questions implemented in Python, experience-based questions (including success metrics), live coding, and search capabilities for finding models or data needed to solve a problem.",
                            },
                            ...listMessage.map((msg) => ({
                                role: msg.role as
                                    | "user"
                                    | "assistant"
                                    | "system",
                                content: msg.content,
                            })),
                        ],
                        store: true,
                    })
                    .then((res) => {
                        setState(ChatState.USER_TURN);
                        setListMessage((prev) => [
                            ...prev,
                            {
                                role: res.choices[0].message.role,
                                content: res.choices[0].message.content ?? "",
                            },
                        ]);
                    });
                openai.chat.completions
                    .create({
                        model: "gpt-4o-mini",
                        messages: [
                            {
                                role: "system",
                                content:
                                    "You are an English teacher. Your mission is correct the user's message. Always return in function call format.",
                            },
                            ...listMessage.map((msg) => ({
                                role: msg.role as
                                    | "user"
                                    | "assistant"
                                    | "system",
                                content: msg.content,
                            })),
                        ],
                        tools: tools.map((tool) => ({
                            ...tool,
                            type: "function",
                        })),
                        store: true,
                    })
                    .then((res) => {
                        const result = JSON.parse(
                            res.choices?.[0]?.message?.tool_calls?.[0]?.function
                                .arguments ?? ""
                        );
                        setCorrectMessage(result["correction"]);
                        console.log(result["correct"]);
                        console.log(res);
                        setIsCorrectMessage(result["correct"]);
                    });

                // eslint-disable-next-line @typescript-eslint/no-unused-vars
            } catch (error) {
                message.error("Error when sending message, Checking your API");
                // You can also handle the error further here, e.g. show a notification to the user
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [conversationId, listMessage]);

    const { messagesRef, isAtBottom, scrollToBottom } = useScrollAnchor();
    useEffect(() => {
        setNewChatId(conversationId);
    }, [conversationId, setNewChatId]);
    return (
        <div
            className="bg-background group w-full overflow-auto flex h-fit"
            ref={listRef}
        >
            <div className="mx-auto justify-center flex max-w-3xl w-full">
                <div
                    className={cn("w-full")}
                    style={{ marginBottom: `${height + 80}px` }}
                    ref={messagesRef}
                >
                    {listMessage?.length ? (
                        <ChatList
                            messages={listMessage}
                            messageCorrection={correctMessage}
                            isCorrectMessage={isCorrectMessage ?? true}
                        />
                    ) : (
                        <div className="flex flex-col justify-center gap-6">
                            <EmptyScreen />
                        </div>
                    )}
                </div>
            </div>
            <div
                ref={chatPanelRef}
                className="bottom-0 fixed w-full pb-6 bg-background"
            >
                <ChatPanel
                    setListMessage={setListMessage}
                    isAtBottom={isAtBottom}
                    scrollToBottom={scrollToBottom}
                />
            </div>
        </div>
    );
}
