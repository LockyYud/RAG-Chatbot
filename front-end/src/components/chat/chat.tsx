"use client";

import { useLocalStorage } from "@/libs/hooks/use-local-storage";
import { useEffect, useRef, useState } from "react";
import { ChatList } from "./chat-list";
import { ChatState, Message } from "@/libs/types";
import { ChatPanel } from "./chat-panel";
import { useScrollAnchor } from "@/libs/hooks/use-scroll-anchor";
import { cn } from "@/libs/utils";
import { EmptyScreen } from "./empty-screen";
import { sendMessages } from "@/api/chat-api";
import { useChatContext } from "@/libs/context/chat-context";
import { message } from "antd";
import ScrollToBottomButton from "./button-to-end";

interface InforConversation {
    grade: number;
    messages: Array<Message>;
    topic: string;
}

export function Chat() {
    const { setState, id: conversationId, fileList } = useChatContext();
    const [listMessage, setListMessage] = useState<Message[]>([]);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_, setNewChatId] = useLocalStorage("newChatId", conversationId);
    const chatPanelRef = useRef<HTMLDivElement>(null);
    const [height, setHeight] = useState<number>(0);
    const listRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (listRef.current) {
            listRef.current.scrollTop = window.innerHeight;
        }
        window.scrollTo(0, document.body.scrollHeight);
    }, [listMessage]);
    useEffect(() => {
        if (chatPanelRef.current) {
            const resizeObserver = new ResizeObserver((entries) => {
                for (let entry of entries) {
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
                sendMessages(
                    listMessage,
                    fileList,
                    localStorage.getItem("api_base_url")
                ).then((res) => {
                    setState(ChatState.USER_TURN);
                    setListMessage((prev) => [...prev, res]);
                });
            } catch (error) {
                message.error("Error when sending message, Checking your API");
                // You can also handle the error further here, e.g. show a notification to the user
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [conversationId, listMessage]);

    const { messagesRef, scrollRef, isAtBottom, scrollToBottom } =
        useScrollAnchor();
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
                        <>
                            <ChatList
                                messages={listMessage}
                                setMessages={setListMessage}
                            />
                            {/* <div
                                className="fixed"
                                style={{ marginBottom: `${height + 80}px` }}
                            >
                                <ScrollToBottomButton listRef={listRef} />
                            </div> */}
                        </>
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
