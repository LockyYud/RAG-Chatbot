"use client";
import { ChatState, Message } from "@/libs/types";
import { cn } from "@/libs/utils";
import { Divider } from "antd";
import { BotMessage, Suggestion, UserMessage } from "./message";
import { Dispatch, SetStateAction, useEffect, useState } from "react";
import { useChatContext } from "@/libs/context/chat-context";
// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { AxiosError } from "axios";
// import { AxiosError } from 'axios';
export interface ChatList {
    messages: Message[];
    setMessages: Dispatch<SetStateAction<Message[]>>;
}
export function ChatList({ messages, setMessages }: ChatList) {
    const { state } = useChatContext();
    const [suggestion, setSuggestion] = useState<Message[]>([]);
    useEffect(() => {
        if (state == ChatState.BOT_TURN) {
            setSuggestion(() => []);
        }
    }, [state]);
    if (!messages.length) return null;
    return (
        <div className="flex flex-col gap-6">
            {messages.map((message, index) => (
                <div
                    key={index}
                    className={cn(
                        "flex flex-col justify-center",
                        message.role === "user" ? "ml-auto" : ""
                    )}
                >
                    {(message.role === "user" && (
                        <UserMessage>{message.content}</UserMessage>
                    )) || <BotMessage>{message.content}</BotMessage>}
                </div>
            ))}
            {state == ChatState.BOT_TURN && (
                <div>
                    <Divider className="my-4" />
                    <BotMessage>waiting ...</BotMessage>
                </div>
            )}
            <Suggestion
                suggestMessage={suggestion}
                setSuggestion={setSuggestion}
                setListMessage={setMessages}
            />
        </div>
    );
}
