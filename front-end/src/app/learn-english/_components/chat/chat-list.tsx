"use client";
import { ChatState, Message } from "@/libs/types";
import { cn } from "@/libs/utils";
import { List, Spin } from "antd";
import { BotMessage, UserMessage } from "./message";
import { useChatContext } from "@/libs/context/chat-context";
import { LoadingOutlined } from "@ant-design/icons";
import { CorrectionMessage } from "./list-context";

export interface ChatListProps {
    messages: Message[];
    messageCorrection: string;
    isCorrectMessage: boolean;
}

export function ChatList({
    messages,
    messageCorrection,
    isCorrectMessage,
}: Readonly<ChatListProps>) {
    const { state } = useChatContext();

    // Auto scroll to bottom when new messages are added

    if (!messages.length) return null;

    return (
        <div
            style={{
                overflowY: "auto", // Enable vertical scrolling
                borderRadius: "8px",
            }}
            className="mt-16"
        >
            <List
                itemLayout="horizontal"
                dataSource={messages}
                split={false}
                renderItem={(item) => (
                    <List.Item key={item.role} className={cn("flex flex-col")}>
                        <List.Item.Meta />
                        {(item.role === "user" && (
                            <UserMessage>{item.content}</UserMessage>
                        )) || <BotMessage>{item.content}</BotMessage>}
                    </List.Item>
                )}
            />
            {state === ChatState.USER_TURN && (
                <CorrectionMessage
                    messageCorrection={messageCorrection}
                    correct={isCorrectMessage}
                />
            )}
            {state === ChatState.BOT_TURN && (
                <BotMessage>
                    <Spin indicator={<LoadingOutlined spin />} size="large" />
                </BotMessage>
            )}
        </div>
    );
}
