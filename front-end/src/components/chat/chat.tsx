"use client";

import { useLocalStorage } from "@/libs/hooks/use-local-storage";
import { useEffect, useState } from "react";
import { ChatList } from "./chat-list";
import { ChatState, Message } from "@/libs/types";
import { ChatPanel } from "./chat-panel";
import { useScrollAnchor } from "@/libs/hooks/use-scroll-anchor";
import { cn } from "@/libs/utils";
import { EmptyScreen } from "./empty-screen";
import { sendMessages } from "@/api/chat-api";
import { useChatContext } from "@/libs/context/chat-context";

interface InforConversation {
    grade: number;
    messages: Array<Message>;
    topic: string;
}

export function Chat() {
    const { setState, id: conversationId } = useChatContext();
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [InforConversation, setInforConversation] =
        useState<InforConversation>({
            grade: 0,
            messages: [],
            topic: "",
        });

    const [listMessage, setListMessage] = useState<Message[]>([]);
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_, setNewChatId] = useLocalStorage("newChatId", conversationId);
    useEffect(() => {
        if (
            listMessage.at(-1) &&
            listMessage.at(-1)?.role == "user" &&
            localStorage.getItem("api_base_url")
        ) {
            sendMessages(
                listMessage,
                localStorage.getItem("api_base_url")
            ).then((res) => {
                setState(ChatState.USER_TURN);
                setListMessage((prev) => [...prev, res]);
            });
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
            className="bg-background group w-full overflow-auto pl-0"
            ref={scrollRef}
        >
            <div className="mx-auto justify-center flex">
                <div
                    className={cn(
                        "pb-[200px] pt-4 md:pt-10 md:max-w-2xl w-full"
                    )}
                    ref={messagesRef}
                >
                    {listMessage?.length ? (
                        <ChatList
                            messages={listMessage}
                            setMessages={setListMessage}
                        />
                    ) : (
                        <div className="flex flex-col justify-center gap-6">
                            <EmptyScreen />
                        </div>
                    )}
                </div>
            </div>
            <ChatPanel
                setListMessage={setListMessage}
                isAtBottom={isAtBottom}
                scrollToBottom={scrollToBottom}
            />
        </div>
    );
}
